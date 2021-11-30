import pandas as pd
import torchvision.transforms as transforms
from dataset.cocodataset import CoCoDataset
from dataset.pimdataset import AttributeDataset
from utils.cutout import SLCutoutPIL
from randaugment import RandAugment
import os.path as osp
from sklearn.model_selection import train_test_split

def get_datasets(args):
    if args.orid_norm:
        normalize = transforms.Normalize(mean=[0, 0, 0],
                                         std=[1, 1, 1])
        # print("mean=[0, 0, 0], std=[1, 1, 1]")
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        # print("mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]")

    train_data_transform_list = [transforms.Resize((args.img_size, args.img_size)),
                                 RandAugment(),
                                 transforms.ToTensor(),
                                 normalize]
    if args.cutout:
        print("Using Cutout!!!")
        train_data_transform_list.insert(1, SLCutoutPIL(n_holes=args.n_holes, length=args.length))

    test_data_transform_list = [transforms.Resize((args.img_size, args.img_size)),
                                transforms.ToTensor(),
                                normalize]

    if args.dataname == "pim":
        train_data_transform_list.insert(0, transforms.ToPILImage())
        test_data_transform_list.insert(0, transforms.ToPILImage())

    test_data_transform = transforms.Compose(test_data_transform_list)
    train_data_transform = transforms.Compose(train_data_transform_list)


    if args.dataname == 'coco' or args.dataname == 'coco14':
        # ! config your data path here.
        dataset_dir = args.dataset_dir
        train_dataset = CoCoDataset(
            image_dir=osp.join(dataset_dir, 'train2014'),
            anno_path=osp.join(dataset_dir, 'annotations/instances_train2014.json'),
            input_transform=train_data_transform,
            labels_path='data/coco/train_label_vectors_coco14.npy',
            keep_only=args.keep_only
        )
        val_dataset = CoCoDataset(
            image_dir=osp.join(dataset_dir, 'val2014'),
            anno_path=osp.join(dataset_dir, 'annotations/instances_val2014.json'),
            input_transform=test_data_transform,
            labels_path='data/coco/val_label_vectors_coco14.npy',
            keep_only=args.keep_only
        )
    elif args.dataname == 'pim':
        dataset_dir = args.dataset_dir
        df = pd.read_csv(dataset_dir, dtype={"castor": str}, index_col=0)

        X_train, X_test, y_train, y_test = train_test_split(
            df.drop(args.target_col, axis=1),
            df[args.target_col],
            test_size=0.1,
            stratify=df[args.target_col] if not args.is_multilabel else None,
        )

        X_train = X_train.reset_index()
        y_train = y_train.reset_index(drop=True)
        print(y_train)
        X_test = X_test.reset_index()
        y_test = y_test.reset_index(drop=True)
        print(y_test)

        train_dataset = AttributeDataset(
            castors=X_train["castor"],
            labels=y_train,
            inference=False,
            multilabel=args.is_multilabel,
            n_classes=args.num_class,
            transform=train_data_transform
        )
        val_dataset = AttributeDataset(
            castors=X_test["castor"],
            labels=y_test,
            inference=False,
            multilabel=args.is_multilabel,
            n_classes=args.num_class,
            transform=test_data_transform
        )
    else:
        raise NotImplementedError("Unknown dataname %s" % args.dataname)

    print("len(train_dataset):", len(train_dataset))
    print("len(val_dataset):", len(val_dataset))
    return train_dataset, val_dataset
