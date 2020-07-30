from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import argparse

import cv2
import torch
from torch.autograd import Variable
import numpy as np
from glob import glob
from imutils.video import FPS
from tqdm import tqdm

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker
from updatenet.updatenet import UpdateResNet

torch.set_num_threads(1)

parser = argparse.ArgumentParser(description='tracking demo')
parser.add_argument('--config', type=str, default='experiments/siammaske_r50_l3/config.yaml', help='config file')
parser.add_argument('--snapshot', type=str, default='experiments/siammaske_r50_l3/model.pth', help='model name')
parser.add_argument('--video_name', type=str, default='../updatenet/images', help='videos or image files')
parser.add_argument('--roi', type=tuple, default=(631,315,128,123), help='(x, y, w, h')
args = parser.parse_args()


def get_frames(video_name):
    if not video_name:
        cap = cv2.VideoCapture(0)
        # warmup
        for i in range(5):
            cap.read()
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    elif video_name.endswith('avi') or \
        video_name.endswith('mp4'):
        cap = cv2.VideoCapture(args.video_name)
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    else:
        images = glob(os.path.join(video_name, '*.jp*'))
        images = sorted(images,
                        key=lambda x: int(x.split('/')[-1].split('.')[0]))
        for img in images:
            frame = cv2.imread(img)
            yield frame


def main():
    # load config
    cfg.merge_from_file(args.config)
    cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
    device = torch.device('cuda' if cfg.CUDA else 'cpu')

    # create model
    model = ModelBuilder()

    # load model
    model.load_state_dict(torch.load(args.snapshot,
        map_location=lambda storage, loc: storage.cpu()))
    model.eval().to(device)
    updatenet = UpdateResNet()    
    update_model=torch.load('updatenet/checkpoint_lr46_30.pth.tar')['state_dict']
    updatenet.load_state_dict(update_model)
    updatenet.eval().to(device)

    # build tracker
    tracker = build_tracker(model)

    first_frame = True
    zf_pre = None
    if args.video_name:
        video_name = args.video_name.split('/')[-1].split('.')[0]
    else:
        video_name = 'webcam'
    # cv2.namedWindow(video_name, cv2.WND_PROP_FULLSCREEN)
    for num, frame in tqdm(enumerate(get_frames(args.video_name))):
        if first_frame:
            try:
                # init_rect = cv2.selectROI(video_name, frame, False, False)
                init_rect = args.roi
            except:
                exit()
            tracker.init(frame, init_rect)
            zf_pre = tracker.model.zf.cpu().data
            first_frame = False
            fps_cal = FPS().start()
        else:
            outputs = tracker.track(frame)
            update_1 = torch.cat((Variable(tracker.model.zf).cuda(), Variable(zf_pre).cuda(), outputs['zf_cur']), 1)
            update_2 = Variable(tracker.model.zf).cuda()
            zf_new = updatenet(update_1, update_2)
            tracker.model.zf = zf_new
            zf_pre = outputs['zf_cur'].cpu().data
            if 'polygon' in outputs:
                polygon = np.array(outputs['polygon']).astype(np.int32)
                cv2.polylines(frame, [polygon.reshape((-1, 1, 2))],
                              True, (0, 255, 0), 3)
                mask = ((outputs['mask'] > cfg.TRACK.MASK_THERSHOLD) * 255)
                mask = mask.astype(np.uint8)
                frame[:,:,2] = (mask > 0) * 255 *0.75 + (mask == 0) * frame[:,:,2]
            else:
                bbox = list(map(int, outputs['bbox']))
                cv2.rectangle(frame, (bbox[0], bbox[1]),
                              (bbox[0]+bbox[2], bbox[1]+bbox[3]),
                              (0, 255, 0), 3)
            fps_cal.update()
            fps_cal.stop()
            fps_text = 'FPS: {:.2f}'.format(fps_cal.fps())
            cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imwrite('output/{:04d}.jpg'.format(num), frame)


if __name__ == '__main__':
    main()
