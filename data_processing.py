import os.path as osp
import os
import random
import shutil
import pickle

from tqdm import tqdm
from glob import glob
from PIL import Image, ImageFilter
from collections import defaultdict

dataset_dir = 'datasets'


def data_split():
    os.makedirs(dataset_dir, exist_ok=True)
    target_dir = 'CelebAMask-HQ'

    img_list = glob(osp.join(target_dir, 'CelebA-HQ-img', '*.jpg'))

    random.shuffle(img_list)
    train_cut = int(len(img_list) * 0.9)

    train_data = img_list[:train_cut]
    test_data = img_list[train_cut:]

    for mode, data_list in [['train', train_data], ['test', test_data]]:
        print(f'{mode} data processing...')
        mode_dir = osp.join(dataset_dir, mode)
        os.makedirs(mode_dir, exist_ok=True)
        for image_path in tqdm(data_list):
            dst = osp.join(mode_dir, osp.basename(image_path))
            shutil.copy(src=image_path, dst=dst)


def get_color_domain_data(ftimes=100):

    img_list = glob(osp.join(dataset_dir, '*', '*.jpg'))
    img_list = [x for x in img_list if '_' not in osp.basename(x)]
    check_dict = defaultdict(list)

    for img_path in tqdm(img_list):
        img_name = osp.basename(img_path)

        if '_' in img_name:
            continue
        elif osp.exists(osp.join(osp.dirname(img_path), img_name.replace('.jpg', '_color.jpg'))):
            continue

        img_dir = osp.dirname(img_path)
        img = Image.open(img_path)
        for _ in range(ftimes):
            img = img.filter(ImageFilter.MedianFilter(size=3))
        median_path = osp.join(img_dir, img_name.replace('.jpg', '_median.jpg'))
        img.save(median_path)
        failure_dict = get_color_map(median_path)

        check_dict.update(failure_dict)

    print(f'num of failure face : {len(check_dict)}')

    with open('failure_data.plk','wb') as fp:
        pickle.dump(check_dict, fp)


def median_testing(times=300):
    img = Image.open("sample/0.jpg")
    for _ in range(times):
        img = img.filter(ImageFilter.MedianFilter(size=3))
    img.save(f"sample/0_{times}_median.jpg")
    img.close()


def get_colors(target_img, source_list):
    result_list = list()
    for x, y in source_list:
        rgb = target_img.getpixel((x, y))
        result_list.append(rgb)
    return result_list


def get_color_map(img_path, imsize=(512, 512)):

    img_name = osp.basename(img_path)
    dir_name = osp.dirname(img_path)
    fid = str(img_name.split('_')[0]).zfill(5)
    seg_list = ['hair', 'skin', 'l_brow', 'l_eye', 'l_lip', 'mouth', 'neck', 'nose', 'r_brow', 'r_eye', 'u_lip']
    face_im = Image.open(img_path).resize(imsize).convert('RGB')
    color_image = Image.new('RGB', imsize)
    failure_dict = defaultdict(list)

    def find_path(img_name):
        for i in range(0, 15):
            path = osp.join('CelebAMask-HQ', 'CelebAMask-HQ-mask-anno', f'{str(i)}', img_name)
            if osp.exists(path):
                return path
        return False

    for seg in seg_list:

        path = find_path(f'{fid}_{seg}.png')

        if path is False:
            #print(f'{fid}_{seg}.png')
            failure_dict[fid].append(seg)
            #raise FileNotFoundError
            continue

        seg_map = Image.open(path).convert('1')
        area_list = list()

        for i in range(imsize[0]):
            for j in range(imsize[1]):
                if seg_map.getpixel((i, j)) == 255:
                    area_list.append((i, j))

        seg_colors = get_colors(face_im, area_list)

        if len(seg_colors) == 0:
            failure_dict[fid].append(seg)
            continue

        r = sum([x[0] for x in seg_colors]) / len(seg_colors)
        g = sum([x[1] for x in seg_colors]) / len(seg_colors)
        b = sum([x[2] for x in seg_colors]) / len(seg_colors)
        r, g, b = int(r), int(g), int(b)

        for x, y in area_list:
            color_image.putpixel((x, y), (r, g, b))
        seg_map.close()

    face_im.close()
    color_image.save(osp.join(dir_name, f'{int(fid)}_color.jpg'))
    color_image.close()

    return failure_dict


def remove_data(delete=False):

    for m in ['train', 'test']:
        img_list = glob(osp.join(dataset_dir, m, '*.jpg'))
        face_id = set([osp.basename(x).replace('.jpg', '').split('.')[0] for x in img_list])
        verified_id, del_id = set(), set()

        for f_id in face_id:

            origin = osp.exists(osp.join(dataset_dir, m, f'{f_id}.jpg'))
            hed = osp.exists(osp.join(dataset_dir, m, f'{f_id}_hed.jpg'))
            color = osp.exists(osp.join(dataset_dir, m, f'{f_id}_color.jpg'))
            median = osp.exists(osp.join(dataset_dir, m, f'{f_id}_median.jpg'))

            if all([origin, hed, color, median]):
                verified_id.add(f_id)
            else:
                del_id.add(f_id)

        print(f'{m} verified_id : {len(verified_id)}')

        if delete:
            for f_id in del_id:
                for s in ['', '_hed', '_color', '_median']:
                    path = osp.join(dataset_dir, m, f'{f_id}{s}.jpg')
                    if osp.exists(path):
                        os.remove(path)


if __name__ == '__main__':
    random.seed(1234)
    #median_testing()
    #data_split()
    #get_color_domain_data()
    remove_data(delete=False)
