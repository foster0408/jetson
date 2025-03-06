import jetson_inference
import jetson_utils

net = jetson_inference.detectNet("ssd-mobilenet", threshold=0.5)
camera = jetson_utils.gstCamera(640, 480, "/dev/video0")
display = jetson_utils.glDisplay()

while display.IsOpen():
    img, width, height = camera.CaptureRGBA()
    detections = net.Detect(img, width, height)
    display.RenderOnce(img, width, height)
