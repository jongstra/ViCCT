import io
import datetime
from pytz import timezone
# from tqdm.notebook import tqdm
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import cm
import models.ViCCT_models
import models.Swin_ViCCT_models
from timm.models import create_model
from datasets.dataset_utils import img_equal_split, img_equal_unsplit
import torchvision.transforms as standard_transforms
import gradio as gr


# Switch Matplotlib backend.
plt.switch_backend('agg')

# Set some global variables. Only for hardcore users, no need to modify these.
mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Mean and std.dev. of ImageNet
overlap = 32  # We ensure crops have at least this many pixels of overlap.
ignore_buffer = 16  # When reconstructing the whole density map, ignore this many pixels on crop prediction borders.

train_img_transform = standard_transforms.Compose([
    standard_transforms.ToTensor(),
    standard_transforms.Normalize(*mean_std)
])


def load_model(model_name='Swin_ViCCT_large_22k',
               weights_path='models/trained_models/Swin_ViCCT_large_22k_generic_1600_epochs.pth',
               use_cuda="True"):
    """ Creates the ViCCT model and initialises it with the specified weights. """

    model = create_model(  # From the timm library. This function created the model specific architecture.
        model_name,
        init_path=weights_path,
        pretrained_cc=True,
        drop_rate=None if 'Swin' in model_name else 0.,  # Dropout

        # Bamboozled by Facebook. This isn't drop_path_rate, but rather 'drop_connect'.
        # I'm not yet sure what it is for the Swin version
        drop_path_rate=None if 'Swin' in model_name else 0.,
        drop_block_rate=None,  # Drops our entire Transformer blocks I think? Not used for ViCCT.
    )

    if use_cuda:
        model = model.cuda()  # Place model on GPU

    model = model.eval()

    return model


def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    return img


def create_density_map_image(den, pred_cnt):
    """Create a density map image using the density map."""

    fig = plt.figure(figsize=(1440 / 100, 810 / 100), dpi=100)
    plt.title(f'Predicted count: {pred_cnt:.1f}')
    plt.imshow(den, cmap=cm.jet)
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    den_im = fig2img(fig)

    # Clean up memory.
    fig.clear()
    plt.close(fig)

    return den_im


def create_overlay_image(input_image, den, pred_cnt):
    """Use an image and its generated density map + prediction count to create & return an overlayed image."""

    img_heat = np.array(input_image)
    den_heat = den.clone().numpy()

    den_heat = den_heat / 3000  # Scale values to original domain
    den_heat[den_heat < 0] = 0  # Remove negative values
    den_heat = den_heat / den_heat.max()  # Normalise between 0 and 1

    den_heat **= 0.5  # Reduce large values, increase small values
    den_heat *= 255  # Values from 0 to 255 now
    den_heat[den_heat < 50] = 0  # Threshold of 50

    img_heat[:, :, 0][den_heat > 0] = img_heat[:, :, 0][den_heat > 0] / 2
    img_heat[:, :, 1][den_heat > 0] = img_heat[:, :, 1][den_heat > 0] / 2
    img_heat[:, :, 2][den_heat > 0] = den_heat[den_heat > 0]

    fig = plt.figure(figsize=(1440 / 100, 810 / 100), dpi=100)
    plt.title(f'Predicted count: {pred_cnt:.1f}')
    plt.imshow(img_heat, cmap=cm.jet)

    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    overlay_im = fig2img(fig)

    # Clean up memory.
    fig.clear()
    plt.close(fig)

    return overlay_im


def process_image(img_stack, pred_stack, img_h, img_w, use_cuda=True):
    """Process a prepared image (using its prepared elements) with the ViCCT model."""

    if not use_cuda and img_stack.shape[0] > 100:  # If on CPU and more than 100 image crops.
        print('\033[93m'
              'WARNING: you are making a prediction on a very large image. This might take a long time! '
              'You may want to use a lower "Scale Factor" value for faster processing. '
              'You can stop a running process by pressing F5.'
              '\033[0m')

    with torch.no_grad():  # Dont make gradients
        print(f"Processing {len(img_stack)} image parts.")
        for idx, img_crop in enumerate(img_stack):  # For each image crop
            pred_stack[idx] = model.forward(img_crop.unsqueeze(0)).cpu()  # Make prediction.
    print('Done!')

    # Unsplit the perdiction crops to get the entire density map of the image.
    den = img_equal_unsplit(pred_stack, overlap, ignore_buffer, img_h, img_w, 1)
    den = den.squeeze()  # Remove the channel dimension

    # Compute the perdicted count, which is the sum of the entire density map. Note that the model is trained with density maps
    # scaled by a factor of 3000 (See sec 5.2 of my thesis for why: https://scripties.uba.uva.nl/search?id=723178). In short,
    # This works :)
    pred_cnt = den.sum() / 3000

    return den, pred_cnt


def prepare_loaded_image(img, use_cuda=True):
    """Prepare an image for processing with the ViCCT model."""

    # Get image dimensions
    img_w, img_h = img.size

    # Before we make the prediction, we normalise the image and split it up into crops
    img = train_img_transform(img)
    img_stack = img_equal_split(img, 224,
                                overlap)  # Split the image ensuring a minimum of 'overlap' of overlap between crops.

    if use_cuda:
        img_stack = img_stack.cuda()  # Place image stack on GPU

    # This is the placeholder where we store the model predictions.
    pred_stack = torch.zeros(img_stack.shape[0], 1, 224, 224)

    return img_stack, pred_stack, img_h, img_w


def rescale_image(img, scale_factor):
    """Rescale and return an image based on the given scale factor."""

    # Get image dimensions
    img_w, img_h = img.size

    # Rescale image
    if scale_factor != 1.:
        new_w, new_h = round(img_w * scale_factor), round(img_h * scale_factor)
        img = img.resize((new_w, new_h))

    return img


def compute_scale_factor(image, ideal_min_res=2000):
    """Computes the scale factor for images."""

    # Get image resolution.
    x_res, y_res = image.size

    # Get a downscale factor if the image is very large.
    min_res = min(x_res, y_res)
    if min_res < ideal_min_res:
        factor = 1
    if min_res > ideal_min_res:
        factor = ideal_min_res / min_res
        if factor > 1:
            factor = 1

    # Get an upscale factor if the image is very large.
    if min_res < 224:
        factor = 240 / min_res  # Aim to upscale the image to be a bit larger than 224px for its smallest axis.

    return factor


def normalize_image_orientation(img):
    """Modifies image to its normalized orientation/rotation using exif information. Returns normalized image."""

    # Get image orientation from exit (return unchanged image if exif or rotation data is not available).
    try:
        exif = img.getexif()
        orientation = dict(exif.items())[274]  # 274 is the exif key for image orientation.
    except (KeyError, AttributeError) as e:
        return img

    # Rotate image to normal orientation.
    if orientation == 2:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    elif orientation == 3:
        img = img.rotate(180)
    elif orientation == 4:
        img = img.rotate(180).transpose(Image.FLIP_LEFT_RIGHT)
    elif orientation == 5:
        img = img.rotate(-90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
    elif orientation == 6:
        img = img.rotate(-90, expand=True)
    elif orientation == 7:
        img = img.rotate(90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
    elif orientation == 8:
        img = img.rotate(90, expand=True)

    return img


def count_people(image_input):
    """Count the amount of people in an image. Return the resulting density map image, overlay image, and count."""

    # Catch situation where input image is of type None.
    if not image_input:
        return None, None, None

    # Set start time.
    t0 = datetime.datetime.now().astimezone(timezone('Europe/Amsterdam'))

    # Normalize image orientation.
    image_input = normalize_image_orientation(image_input)

    # Rescale image.
    scale_factor = compute_scale_factor(image_input)
    image = rescale_image(image_input, scale_factor)

    # Give the user an error for images with a too low resolution. (alternative: upscale)
    w = image.width
    h = image.height
    if w < 224 or h < 224:
        raise gr.Error("Image is too small, please provide a bigger image (244x244 or larger) and try again.")
        return None, None, 0

    # Prepare and process image (create prediction).
    img_stack, pred_stack, img_h, img_w = prepare_loaded_image(image)
    den, pred_cnt = process_image(img_stack, pred_stack, img_h, img_w)

    # Create density map image.
    den_im = create_density_map_image(den, pred_cnt)

    # Create overlay image.
    overlay_im = create_overlay_image(image, den, pred_cnt)

    # Log succesful counting.
    t1 = datetime.datetime.now().astimezone(timezone('Europe/Amsterdam'))
    processing_time = (t1 - t0).total_seconds()
    with open("log.txt", "a") as myfile:
        myfile.write(
            f"{t1}; succesfully processed an image of size {w}*{h} (w*h) -after possible downscaling- in {processing_time} seconds.\n")

    return den_im, overlay_im, round(float(pred_cnt), 1)


# Launch the demo website.
def launch_demo():
    demo = gr.Blocks(title="Crowd Counter")

    with demo:
        # Introduction.
        gr.Markdown("# Amsterdam Crowd Counter")
        gr.Markdown(
            "Upload an image & count people. Processing should take below 40 seconds. If not, refresh the page (F5).")

        # Interactive elements.
        image_input = gr.Image(type='pil')
        count_button = gr.Button("Count People")
        count_result = gr.Number(label="People Count", elem_id='count', visible=False)

        with gr.Row():
            with gr.Column():
                image_output_overlay = gr.Image(elem_id='output_image', interactive=False)
            with gr.Column():
                image_output = gr.Image(elem_id='output_image', interactive=False)

        # Interactions.
        count_button.click(fn=count_people, inputs=image_input,
                           outputs=[image_output, image_output_overlay, count_result])

        # Explanation about this website/service.
        gr.Markdown("""Counting results are generated using an AI model called [ViCCT](https://github.com/jongstra/ViCCT).
                       This model is trained using multiple annotated datasets with large amounts of crowds.
                       The resulting model is only usable for counting people and estimating crowd densities,
                       not for identifying individuals.""")
        gr.Markdown("""This service is in testing phase and is provided "as-is",
                       without warranty of any kind, nor any guarantees about correctness of results.
                       Uploaded images are only processed on our server, not saved to disk.
                       This service should never be used as a sole means of crowd size estimation,
                       but is intended to be used for human-assisted solutions.""")
        gr.Markdown(
            "For questions/feedback, contact us at [crowdcounter@amsterdam.nl](mailto:crowdcounter@amsterdam.nl).")

    # demo.launch(share=False)
    demo.launch(server_port=7860, share=False)


if __name__ == "__main__":
    gr.close_all()  # Try to close any running Gradio processes, to free up ports.
    global model
    model = load_model()
    launch_demo()