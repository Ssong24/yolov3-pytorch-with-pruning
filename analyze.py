import argparse  # user argument

import torch.distributed as dist
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import test  # import test.py to get mAP after each epoch
from models import *
from utils.datasets import *
from utils.utils import *
from utils.radam import *      # Song's modified code
from sparsity import updateBN  # Channel Pruning

mixed_precision = True
try:  # Mixed precision training
    from apex import amp
except:
    mixed_precision = False  # not installed

hyp = {'giou': 1.77, # 3.54,  # giou loss gain
       'cls': 18.7, # 37.4,  # cls loss gain
       'cls_pw': 1.0,  # cls BCELoss positive_weight
       'obj': 64.3,  # obj loss gain (*=img_size/320 if img_size != 320)
       'obj_pw': 1.0,  # obj BCELoss positive_weight
       'iou_t': 0.225,  # iou training threshold
       'lr0':  0.0005, #0.001, # 0.01,  # initial learning rate (SGD=5E-3, Adam=5E-4)
       'lrf': -4.,  # final LambdaLR learning rate = lr0 * (10 ** lrf)
       'momentum': 0.937,  # SGD momentum
       'weight_decay': 0.000484,  # optimizer weight decay
       'fl_gamma': 0.0,  # focal loss gamma (efficientDet default is gamma=1.5)
       'hsv_h': 0.0138,  # image HSV-Hue augmentation (fraction)
       'hsv_s': 0.678,  # image HSV-Saturation augmentation (fraction)
       'hsv_v': 0.36,  # image HSV-Value augmentation (fraction)
       'degrees': 1.98 , #* 0,  # image rotation (+/- deg)
       'translate': 0.05,# * 0,  # image translation (+/- fraction)
       'scale': 0.05, # * 0,  # image scale (+/- gain)
       'shear': 0.641, #* 0  # image shear (+/- deg)
       }

def analyze():
    cfg = opt.cfg
    data = opt.data
    img_size, img_size_test = opt.img_size if len(opt.img_size) == 2 else opt.img_size * 2  # train, test sizes
    epochs = opt.epochs  # 500200 batches at bs 64, 117263 images = 273 epochs
    batch_size = opt.batch_size
    accumulate = opt.accumulate  # effective bs = batch_size * accumulate = 16 * 4 = 64
    weights = opt.weights  # initial training weights

    # Initialize
    init_seeds()
    if opt.multi_scale:
        img_sz_min = round(img_size / 32 / 1.5)
        img_sz_max = round(img_size / 32 * 1.5)
        img_size = img_sz_max * 32  # initiate with maximum multi_scale size
        print('Using multi-scale %g - %g' % (img_sz_min * 32, img_size))

    # Configure run
    data_dict = parse_data_cfg(data)
    # check .data's num of classes and n-classes
    assert int(data_dict['classes']) == opt.n_classes, "Different number of classes btw .data and opt.n_classes"

    train_path = data_dict['train']
    test_path = data_dict['valid']
    nc = 1 if opt.single_cls else int(data_dict['classes'])  # number of classes

    # Initialize model
    model = Darknet(cfg).to(device)

    # Optimizer
    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in dict(model.named_parameters()).items():
        if '.bias' in k:
            pg2 += [v]  # biases
        elif 'Conv2d.weight' in k:
            pg1 += [v]  # apply weight_decay
        else:
            pg0 += [v]  # all else

    if opt.adam:
        optimizer = optim.Adam(pg0, lr=hyp['lr0'])
    else:
        optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)
    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    del pg0, pg1, pg2

    # Set start_epoch = 0
    start_epoch = 0
    # Criterion to pick best model
    best_fitness = 0.0

    # Try to download the weight file of opt
    attempt_download(weights)
    if weights.endswith('.pt'):  # pytorch format
        chkpt = torch.load(weights, map_location=device)

        # load model
        try:
            print('[Loading model...]')
            chkpt['model'] = {k: v for k, v in chkpt['model'].items() if model.state_dict()[k].numel() == v.numel()}
            model.load_state_dict(chkpt['model'], strict=False)
        except KeyError as e:
            s = "%s is not compatible with %s. Specify --weights '' or specify a --cfg compatible with %s. " \
                "See https://github.com/ultralytics/yolov3/issues/657" % (opt.weights, opt.cfg, opt.weights)
            raise KeyError(s) from e

        # load optimizer
        if chkpt['optimizer'] is not None:
            print('[Loading optimizer...]')
            optimizer.load_state_dict(chkpt['optimizer'])
            best_fitness = chkpt['best_fitness']

        start_epoch = chkpt['epoch'] + 1
        del chkpt

    elif len(weights) > 0:  # darknet format
        load_darknet_weights(model, weights)

    if mixed_precision:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1', verbosity=0) # verbosity=1: set to 0 to suppress Amp-related output

    lf = lambda x: (((1 + math.cos(x * math.pi / epochs)) / 2) ** 1.0) * 0.95 + 0.05  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # This also has warm-up stage of lr

    # Initialize distributed training
    if device.type != 'cpu' and torch.cuda.device_count() > 1 and torch.distributed.is_available():
        dist.init_process_group(backend='nccl',  # 'distributed backend'
                                init_method='tcp://127.0.0.1:9999',  # distributed training init method
                                world_size=1,  # number of nodes for distributed training
                                rank=0)  # distributed training node rank
        model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)

        model.yolo_layers = model.module.yolo_layers  # move yolo layer indices to top level

    # Dataset
    dataset = LoadImagesAndLabels(train_path, img_size, batch_size, augment=True,
                                  hyp=hyp,  # augmentation hyperparameters
                                  rect=opt.rect,  # rectangular training
                                  cache_images=opt.cache_images,
                                  single_cls=opt.single_cls)
    # Dataloader
    batch_size = min(batch_size, len(dataset))
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=nw,
                                             shuffle=not opt.rect,  # Shuffle=True unless rectangular training is used
                                             pin_memory=True,
                                             collate_fn=dataset.collate_fn)

    # Test loader - valid
    testloader = torch.utils.data.DataLoader(LoadImagesAndLabels(test_path, img_size_test, batch_size * 2,
                                                                 hyp=hyp,
                                                                 rect=True,
                                                                 cache_images=opt.cache_images,
                                                                 single_cls=opt.single_cls),
                                             batch_size=batch_size * 2, num_workers=nw,
                                             pin_memory=True, collate_fn=dataset.collate_fn)

    # Model parameters
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 0.0  # giou loss ratio (obj_loss = 1.0 or giou)
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device)  # attach class weights

    # for i, (md, ml, nm) in enumerate(zip(model.module_defs ,model.module_list, model.named_modules())):
    #     # print('{} {}'.format(i ,m))
    #     # print('{} {}'.format(i, md))
    #     print('{} {}'.format(i,nm))






    # Start training
    n = opt.name # = 45
    # opt.evolve = False
    nb = len(dataloader)  # num of batches
    prebias = start_epoch == 0
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # 'P', 'R', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification'
    t0 = time.time()
    print('num of batches| prebias | start epoch | total epochs| = {}, {}, {}, {}'.format(nb, prebias, start_epoch, epochs))


    for epoch in range(start_epoch, epochs):
        model.train()

        if prebias:
            ne = 3  # number of prebias epochs
            ps = 0.1, 0.9  # prebias settings (lr=0.1, momentum=0.9)
            if epoch == ne:
                ps = hyp['lr0'], hyp['momentum']  # normal training settings
                model.gr = 1.0  # giou loss ratio (obj_loss = giou)
                print_model_biases(model)
                prebias = False

            # Bias optimizer settings
            optimizer.param_groups[2]['lr'] = ps[0]
            if optimizer.param_groups[2].get('momentum') is not None:  # for SGD but not Adam
                optimizer.param_groups[2]['momentum'] = ps[1]

        mloss = torch.zeros(4).to(device)  # mean losses
        print(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'GIoU', 'obj', 'cls', 'total', 'targets', 'img_size'))

        pbar = tqdm(enumerate(dataloader), total=nb)  # progress bar

        # one batch
        for i, (imgs, targets, paths, _) in pbar:
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device).float() / 255.0  # uint8 to float32 --> 0-255 to 0.0-1.0
            targets = targets.to(device)


            n_burn = 200   # number of burn-in batches
            if ni <= n_burn:
                # g = (ni / n_burn) ** 2  # gain
                for x in model.named_modules():  # initial stats may be poor, wait to track
                    if x[0].endswith('BatchNorm2d'):
                        x[1].track_running_stats = ni == n_burn  # ??

            # Multi-Scale training

            # Run model
            pred = model(imgs)
            # len(targets): objects' GT
            # len(pred) = 3
            # pred[0].shape: [bs, 3, 13, 13, 85]
            # pred[1].shape: [bs, 3, 26, 26, 85]
            # pred[2].shape: [bs, 3, 52, 52, 85]

            # Compute loss
            loss, loss_items = compute_loss(pred, targets, model)
            if not torch.isfinite(loss):
                print('WARNING: non-finite loss, ending training ', loss_items)
                return results

            # Scale loss by nominal batch_size of 64
            loss *= batch_size / 64

            # Compute gradient
            if mixed_precision:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            # Optimize accumulated gradient
            if ni % accumulate == 0:
                updateBN(opt.s, model)
                optimizer.step()
                optimizer.zero_grad()

            # Print batch results
            mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
            mem = '%.3gG' % (torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
            s = ('%10s' * 2 + '%10.3g' * 6) % ('%g/%g' % (epoch, epochs - 1), mem, *mloss, len(targets), img_size)
            pbar.set_description(s)  # display func
            # end one batch

            # Display the model_info except Route, Yolo Layer
            # torch_utils.model_info(model, verbose=True)

        # Update scheduler
        scheduler.step()

        final_epoch = epoch + 1 == epochs
        # Calculate mAP when (notest == False) or final_epoch
        if not opt.notest or final_epoch:
            is_coco = any([x in data for x in ['coco.data', 'coco2014.data', 'coco2017.data']]) and model.nc == 80 # False
            results, maps, class_names = test.test(opt,
                                      cfg,
                                      data,
                                      batch_size=batch_size * 2,
                                      img_size=img_size_test,
                                      model=model,
                                      conf_thres=0.001 if final_epoch else 0.01,  # 0.001 for best mAP, 0.01 for speed
                                      iou_thres=0.6,
                                      save_json=final_epoch and is_coco,
                                      single_cls=opt.single_cls,
                                      dataloader=testloader)

        # Write the result of one epoch in result.txt

        # Plot a line-by-line description of PyTorch model

        #
        # Write Tensorboard results

        # Update best mAP --> Use the result of valid dataset
        fi = fitness(np.array(results).reshape(1, -1))  # fitness_i = weighted combination of [P, R, mAP, F1]
        if fi > best_fitness:
            best_fitness = fi

        # Save training results

        # end one epoch

    # end training
    print('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))
    dist.destroy_process_group() if torch.cuda.device_count() > 1 else None
    torch.cuda.empty_cache()

    return results







if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=700)  # 500200 batches at bs 16, 117263 COCO images = 273 epochs
    parser.add_argument('--batch-size', type=int, default=16)  # effective bs = batch_size * accumulate = 16 * 4 = 64
    parser.add_argument('--accumulate', type=int, default=4, help='batches to accumulate before optimizing')
    parser.add_argument('--cfg', type=str, default='input/cfg/yolov3-spp.cfg', help='*.cfg path')
    parser.add_argument('--data', type=str, default='input/dataset/cityscape/cityscape.data', help='*.data path')
    parser.add_argument('--multi-scale', action='store_true', help='adjust (67% - 150%) img_size every 10 batches')
    parser.add_argument('--img-size', nargs='+', type=int, default=[416], help='train and test image-sizes')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', action='store_true', help='resume training from last.pt')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--weights', type=str, default='input/pretrained_weights/yolov3-spp-ultralytics.pt', help='initial weights path')
    parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--adam', action='store_true', help='use adam optimizer')
    parser.add_argument('--single-cls', action='store_true', help='train as single-class dataset')
    parser.add_argument('--var', type=float, help='debug variable')
    parser.add_argument('--s', type=float, default=0.0001, help='scale for sparsity training')
    parser.add_argument('--output', type=str, default='results', help='model result folder')  # output folder
    parser.add_argument('--n-classes', type=int, default=13, help='number of classes in the dataset')
    parser.add_argument('--resume-etri', action='store_true', help='flag for resuming ETRI-pretrained model training')
    opt = parser.parse_args()

    device = torch_utils.select_device(opt.device, apex=mixed_precision, batch_size=opt.batch_size)
    if device.type == 'cpu':
        mixed_precision = False

    analyze()

