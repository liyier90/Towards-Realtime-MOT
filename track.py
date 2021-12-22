import os

os.environ["LOGURU_LEVEL"] = "INFO"

import argparse
import logging
import os.path as osp

import cv2
import motmetrics as mm
import torch

import utils.datasets as datasets
from tracker.multitracker import JDETracker
from utils import visualization as vis
from utils.evaluation import Evaluator
from utils.log import logger
from utils.parse_config import parse_model_cfg
from utils.timer import Timer
from utils.utils import *


def write_results(filename, results, data_type):
    if data_type == "mot":
        save_format = "{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n"
    elif data_type == "kitti":
        save_format = "{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n"
    else:
        raise ValueError(data_type)

    with open(filename, "w") as f:
        for frame_id, tlwhs, track_ids in results:
            if data_type == "kitti":
                frame_id -= 1
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(
                    frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h
                )
                f.write(line)
    logger.info("save results to {}".format(filename))


def eval_seq(
    opt,
    dataloader,
    data_type,
    result_filename,
    save_dir=None,
    show_image=True,
    frame_rate=30,
):
    """
    Processes the video sequence given and provides the output of tracking result (write the results in video file)

    It uses JDE model for getting information about the online targets present.

    Parameters
    ----------
    opt : Namespace
          Contains information passed as commandline arguments.

    dataloader : LoadVideo
                 Instance of LoadVideo class used for fetching the image sequence and associated data.

    data_type : String
                Type of dataset corresponding(similar) to the given video.

    result_filename : String
                      The name(path) of the file for storing results.

    save_dir : String
               Path to the folder for storing the frames containing bounding box information (Result frames).

    show_image : bool
                 Option for shhowing individial frames during run-time.

    frame_rate : int
                 Frame-rate of the given video.

    Returns
    -------
    (Returns are not significant here)
    frame_id : int
               Sequence number of the last sequence
    """

    if save_dir:
        mkdir_if_missing(save_dir)
    tracker = JDETracker(vars(opt), frame_rate=frame_rate)
    timer = Timer()
    results = []
    frame_id = 0
    # img is resized, imo0 is original size
    for path, img, img0 in dataloader:
        if frame_id % 60 == 0:
            logger.info(
                "Processing frame {} ({:.2f} fps)".format(
                    frame_id, 1.0 / max(1e-5, timer.average_time)
                )
            )

        # run tracking
        timer.tic()
        blob = torch.from_numpy(img).cuda().unsqueeze(0)
        online_targets = tracker.update(blob, img0)
        online_tlwhs = []
        online_ids = []
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > opt.min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
        timer.toc()
        # save results
        results.append((frame_id + 1, online_tlwhs, online_ids))
        if show_image or save_dir is not None:
            online_im = vis.plot_tracking(
                img0,
                online_tlwhs,
                online_ids,
                frame_id=frame_id,
                fps=1.0 / timer.average_time,
            )
        if show_image:
            cv2.imshow("online_im", online_im)
        if save_dir is not None:
            cv2.imwrite(
                os.path.join(save_dir, "{:05d}.jpg".format(frame_id)), online_im
            )
        frame_id += 1
    # save results
    write_results(result_filename, results, data_type)
    return frame_id, timer.average_time, timer.calls


def main(
    opt,
    data_root="/data/MOT16/train",
    det_root=None,
    seqs=("MOT16-05",),
    exp_name="demo",
    save_images=False,
    save_videos=False,
    show_image=True,
):
    logger.setLevel(logging.INFO)
    result_root = os.path.join(data_root, "..", "results", exp_name)
    mkdir_if_missing(result_root)
    data_type = "mot"

    # Read config
    cfg_dict = parse_model_cfg(opt.cfg)
    # Set to input size defined by model config
    opt.img_size = [int(cfg_dict[0]["width"]), int(cfg_dict[0]["height"])]

    # run tracking
    accs = []
    n_frame = 0
    timer_avgs, timer_calls = [], []
    for seq in seqs:
        output_dir = (
            os.path.join(data_root, "..", "outputs", exp_name, seq)
            if save_images or save_videos
            else None
        )

        logger.info("start seq: {}".format(seq))
        dataloader = datasets.LoadImages(osp.join(data_root, seq, "img1"), opt.img_size)
        result_filename = os.path.join(result_root, "{}.txt".format(seq))
        meta_info = open(os.path.join(data_root, seq, "seqinfo.ini")).read()
        frame_rate = int(
            meta_info[meta_info.find("frameRate") + 10 : meta_info.find("\nseqLength")]
        )
        nf, ta, tc = eval_seq(
            opt,
            dataloader,
            data_type,
            result_filename,
            save_dir=output_dir,
            show_image=show_image,
            frame_rate=frame_rate,
        )
        n_frame += nf
        timer_avgs.append(ta)
        timer_calls.append(tc)

        # eval
        logger.info("Evaluate seq: {}".format(seq))
        evaluator = Evaluator(data_root, seq, data_type)
        accs.append(evaluator.eval_file(result_filename))
        if save_videos:
            output_video_path = osp.join(output_dir, "{}.mp4".format(seq))
            cmd_str = "ffmpeg -f image2 -i {}/%05d.jpg -c:v copy {}".format(
                output_dir, output_video_path
            )
            os.system(cmd_str)
    timer_avgs = np.asarray(timer_avgs)
    timer_calls = np.asarray(timer_calls)
    all_time = np.dot(timer_avgs, timer_calls)
    avg_time = all_time / np.sum(timer_calls)
    logger.info(
        "Time elapsed: {:.2f} seconds, FPS: {:.2f}".format(all_time, 1.0 / avg_time)
    )

    # get summary
    metrics = mm.metrics.motchallenge_metrics
    mh = mm.metrics.create()
    summary = Evaluator.get_summary(accs, seqs, metrics)
    summary.to_csv(os.path.join(result_root, "eval.csv"))
    strsummary = mm.io.render_summary(
        summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names
    )
    print(strsummary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="track.py")
    parser.add_argument(
        "--cfg",
        type=str,
        default="weights/yolov3_576x320.cfg",
        help="cfg file path",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="weights/jde_576x320_uncertainty.pt",
        help="path to weights file",
    )
    parser.add_argument(
        "--iou-thres",
        type=float,
        default=0.5,
        help="iou threshold required to qualify as detected",
    )
    parser.add_argument(
        "--conf-thres", type=float, default=0.5, help="object confidence threshold"
    )
    parser.add_argument(
        "--nms-thres",
        type=float,
        default=0.4,
        help="iou threshold for non-maximum suppression",
    )
    parser.add_argument(
        "--min-box-area", type=float, default=200, help="filter out tiny boxes"
    )
    parser.add_argument("--track-buffer", type=int, default=30, help="tracking buffer")
    parser.add_argument("--test-mot16", action="store_true", help="tracking buffer")
    parser.add_argument(
        "--save-images", action="store_true", help="save tracking results (image)"
    )
    parser.add_argument(
        "--save-videos", action="store_true", help="save tracking results (video)"
    )
    opt = parser.parse_args()
    print(opt, end="\n\n")

    if not opt.test_mot16:
        seqs_str = """MOT16-02
                      MOT16-04
                      MOT16-05
                      MOT16-09
                      MOT16-10
                      MOT16-11
                      MOT16-13
                    """
        data_root = "/content/MOT16/train"
        # data_root = "/home/yier/Datasets/MOT16-short/train"
    else:
        seqs_str = """MOT16-01
                     MOT16-03
                     MOT16-06
                     MOT16-07
                     MOT16-08
                     MOT16-12
                     MOT16-14"""
        data_root = "/home/wangzd/datasets/MOT/MOT16/images/test"
    seqs = [seq.strip() for seq in seqs_str.split()]

    main(
        opt,
        data_root=data_root,
        seqs=seqs,
        exp_name=opt.weights.split("/")[-2],
        show_image=False,
        save_images=opt.save_images,
        save_videos=opt.save_videos,
    )

#           IDF1   IDP   IDR  Rcll  Prcn  GT MT PT  ML  FP     FN IDs   FM  MOTA  MOTP IDt IDa IDm
# MOT16-02 14.0% 79.9%  7.7%  8.3% 86.6%  54  3  3  48 231  16345  18   42  6.9% 0.213   7   6   2
# MOT16-04 11.4% 91.4%  6.1%  6.3% 95.1%  83  1  5  77 154  44552  10   28  6.0% 0.214   7   4   2
# MOT16-05 12.5% 73.3%  6.8%  8.2% 87.9% 125  3  8 114  77   6261  30   26  6.6% 0.223  15  15   5
# MOT16-09 19.0% 79.7% 10.8% 11.5% 84.4%  25  0  6  19 111   4655   2    4  9.3% 0.164   0   2   0
# MOT16-10 14.3% 74.9%  7.9%  8.9% 84.6%  54  4 11  39 199  11221  57  108  6.8% 0.235  28  19   3
# MOT16-11 19.0% 78.7% 10.8% 12.5% 91.0%  69  5 10  54 113   8031  29   41 10.9% 0.207  11  11   1
# MOT16-13 16.0% 79.0%  8.9% 10.5% 92.6% 107  5 15  87  96  10252  31  105  9.4% 0.228  32  10  12
# OVERALL  13.7% 82.2%  7.5%  8.2% 90.3% 517 21 58 438 981 101317 177  354  7.2% 0.214 100  67  25
