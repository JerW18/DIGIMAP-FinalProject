from flask import Flask, render_template, request, send_file
import cv2
import numpy as np
import io

app = Flask(__name__)

@app.route("/")
def run_app():
    return render_template('index.html')

protoPath = "models\deploy.prototxt"
modelPath = "models\hed_pretrained_bsds.caffemodel"
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

def resize_to_1080p(image):
    # Get the current dimensions
    height, width = image.shape[:2]

    # Define the target width and height for 1080p resolution
    target_width = 1920
    target_height = 1080

    if width > target_width or height > target_height:
        # Calculate the scaling factors
        scale_x = target_width / width
        scale_y = target_height / height

        # Choose the minimum scaling factor to maintain aspect ratio
        scale_factor = min(scale_x, scale_y)

        # Resize the image using interpolation
        image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

    return image

@app.route('/process_image', methods=['POST'])
def process_image():
    # Load the image from the request
    file = request.files['image']
    if file:
        # Read the image file
        nparr = np.fromstring(file.read(), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Resize the image to 1080p if needed
        img = resize_to_1080p(img)

        # Perform image processing
        (H, W) = img.shape[:2]
        mean_pixel_values= np.average(img, axis=(0,1))
        blob = cv2.dnn.blobFromImage(img, scalefactor=0.7, size=(W, H), mean=(mean_pixel_values[0], mean_pixel_values[1], mean_pixel_values[2]))
        net.setInput(blob)
        hed = net.forward()
        hed = hed[0,0,:,:]  # Drop the other axes
        hed = (255 * hed).astype("uint8")  # Rescale to 0-255

        # Encode the image data to a binary format
        img_bytes = cv2.imencode('.jpg', hed)[1].tobytes()

        # Return the image data as a binary response
        return send_file(io.BytesIO(img_bytes),
                         mimetype='image/jpeg',
                         as_attachment=False)

    else:
        return "No image file provided"