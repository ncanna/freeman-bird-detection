# Helper funtions from megaDetector:
# https://github.com/agentmorris/MegaDetector/blob/main/megadetector/detection/video_utils.py

def frames_to_video(images, fs, output_file_name, codec_spec=default_fourcc):
    """
    Given a list of image files and a sample rate, concatenates those images into
    a video and writes to a new video file.

    Args:
        images (list): a list of frame file names to concatenate into a video
        fs (float): the frame rate in fps
        output_file_name (str): the output video file, no checking is performed to make
            sure the extension is compatible with the codec
        codec_spec (str, optional):  codec to use for encoding; h264 is a sensible default
            and generally works on Windows, but when this fails (which is around 50% of the time
            on Linux), mp4v is a good second choice
    """

    if codec_spec is None:
        codec_spec = 'h264'

    if len(images) == 0:
        print('Warning: no frames to render')
        return

    output_dir = os.path.dirname(output_file_name)
    if len(output_dir) > 0:
        os.makedirs(output_dir, exist_ok=True)

    # Determine the width and height from the first image
    frame = cv2.imread(images[0])
    # cv2.imshow('video',frame)
    height, width, channels = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*codec_spec)
    out = cv2.VideoWriter(output_file_name, fourcc, fs, (width, height))

    for image in images:
        frame = cv2.imread(image)
        out.write(frame)

    out.release()
    cv2.destroyAllWindows()