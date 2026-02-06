import ffmpeg
import decord
from skimage.color import gray2rgb
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import random

def save_video(images, path='temp.mp4', crf=18, frame_rate=25):
    # Render the images as the gif:
    height, width, _ = images[0].shape
    out = (
        ffmpeg
        .input('pipe:0', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height))
        .output(path, reset_timestamps=1,
                **{
                    'preset': 'medium',
                    'b:v': '0',
                    'c:v':'libx264',
                    'crf': str(crf),
                    })
        .overwrite_output()
        .run_async(quiet=True, pipe_stdin=True, pipe_stderr=True)
    )
    for frame in images:
        out.stdin.write(frame.tobytes())
    out.stdin.close()
    out.wait()

def get_video_length(name):
    video_reader = decord.VideoReader(name,num_threads=1)
    num_frames = len(video_reader)
    del video_reader
    return num_frames

def load_video_decord(name, 
                      frame_index=None, 
                      num_random=2, 
                      load_full_video=False, 
                      sampling_step=1, 
                      max_frame_cnt=5,
                      is_continuous=False,
                      rotation=False,
                      st_list=None,
                      crop_size=None,
                      ):
    video_reader = decord.VideoReader(name,num_threads=1)
    num_frames = len(video_reader)
    if frame_index is None:
        if load_full_video:
            if sampling_step > 0:
                frame_index = list(range(0, num_frames, sampling_step))[:max_frame_cnt]
            else:
                # get max_frame_cnt indexs from range [0, num_frames) uniformly
                frame_index = np.linspace(0, num_frames, max_frame_cnt + 1, endpoint=False)[1:]
                frame_index = list(np.round(frame_index).astype(np.int32))
        else:
            if is_continuous:
                if st_list is not None:
                    st = np.random.choice(st_list)
                else:
                    st = np.random.randint(0, num_frames - num_random)
                frame_index = list(range(st, st + num_random))
            else:
                if st_list is not None:
                    frame_index = np.random.choice(st_list, replace=False, size=num_random)
                else:
                    frame_index = np.random.choice(num_frames, replace=False, size=num_random)
    video = video_reader.get_batch(frame_index).asnumpy()
    if len(video.shape) == 3:
        video = np.array([gray2rgb(frame) for frame in video])
    if video.shape[-1] == 4:
        video = video[..., :3]
    if rotation:
        video = np.flip(np.transpose(video, (0, 2, 1, 3)), axis=2)
    if crop_size is not None:
        video = center_crop_video(video, crop_size=(crop_size, crop_size))
    del video_reader
    return video, frame_index

def load_video_cv2(name, 
                   frame_index=None, 
                   num_random=2, 
                   load_full_video=False, 
                   sampling_step=1, 
                   max_frame_cnt=5,
                   is_continuous=False,
                   rotation=False,
                   st_list=None,
                   crop_size=None,
                   ):
    """
    Alternative to load_video_decord using OpenCV instead of decord.
    Reads all frames from the video and selects the requested ones.
    """
    import cv2
    
    cap = cv2.VideoCapture(name)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {name}")
    
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Determine frame_index if not provided (same logic as decord version)
    if frame_index is None:
        if load_full_video:
            if sampling_step > 0:
                frame_index = list(range(0, num_frames, sampling_step))[:max_frame_cnt]
            else:
                # get max_frame_cnt indices from range [0, num_frames) uniformly
                frame_index = np.linspace(0, num_frames, max_frame_cnt + 1, endpoint=False)[1:]
                frame_index = list(np.round(frame_index).astype(np.int32))
        else:
            if is_continuous:
                if st_list is not None:
                    st = np.random.choice(st_list)
                else:
                    st = np.random.randint(0, num_frames - num_random)
                frame_index = list(range(st, st + num_random))
            else:
                if st_list is not None:
                    frame_index = np.random.choice(st_list, replace=False, size=num_random)
                else:
                    frame_index = np.random.choice(num_frames, replace=False, size=num_random)
    
    # Read all frames
    frames = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        frame_idx += 1
    
    cap.release()
    
    # Select requested frames
    selected_frames = []
    for idx in frame_index:
        if 0 <= idx < len(frames):
            selected_frames.append(frames[idx])
        else:
            # If index out of range, use the last frame or handle as needed
            selected_frames.append(frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8))
    
    video = np.stack(selected_frames, axis=0)  # (N, H, W, 3)
    
    # Apply same processing as decord version
    if len(video.shape) == 3:
        video = np.array([gray2rgb(frame) for frame in video])
    if video.shape[-1] == 4:
        video = video[..., :3]
    if rotation:
        video = np.flip(np.transpose(video, (0, 2, 1, 3)), axis=2)
    if crop_size is not None:
        video = center_crop_video(video, crop_size=(crop_size, crop_size))
    
    return video, frame_index

def center_crop_video(video, crop_size= (256, 256) ):
    """
    Args:
        video (numpy.ndarray): (num_frames, height, width, channels)
        crop_size (a,b): 256x256
    Returns:
        numpy.ndarray: (num_frames, a, b, channels)
    """
    num_frames, height, width, channels = video.shape

    if height < crop_size[0] or width < crop_size[1]:
        raise ValueError("Video dimensions must be at least the cropped size.")

    start_y = (height - crop_size[0]) // 2
    start_x = (width - crop_size[1]) // 2

    cropped_video = video[:, start_y:start_y + crop_size[0], start_x:start_x + crop_size[1], :]

    return cropped_video