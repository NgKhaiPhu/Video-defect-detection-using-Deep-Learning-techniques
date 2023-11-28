import cv2
import torch
from repo.inference_preprocess import inference_preprocess

def add_text(frame, text, fps, color=(255, 255, 255), thickness=2):
    # add FPS
    fps_position = (50, 50)
    cv2.putText(frame, f"FPS: {fps}", fps_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), thickness, cv2.LINE_AA)
    # add output
    textsize = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 1, 2)[0]
    textX = (frame.shape[1] - textsize[0]) // 2
    cv2.putText(frame, text, (textX, 75), cv2.FONT_HERSHEY_DUPLEX, 2, color, thickness, cv2.LINE_AA)
    
def get_inference_output(frames, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    input_tensor = inference_preprocess(frames)
    with torch.no_grad():
        output = model(input_tensor.to(device))
    rounded_output = torch.round(output)
    return rounded_output            

def inference(input_path, output_path, model):    
    cap = cv2.VideoCapture(input_path)
    if (cap.isOpened() == False): 
      print("Error opening video stream or file")
    
    # Init variables
    num_frames = 16
    cur_frame = 0
    green = (0,255,0)
    red = (0,0,255)
    output_text = 'No flicker'
    output_color = green
    detected_text_delay = 0 

    # Init VideoWriter
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Change the codec as needed
    output_video_writer = cv2.VideoWriter(output_path, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))
    
    while(cap.isOpened()):
        # Take frames for inference
        if cur_frame % num_frames == 0:
            inference_frames = []
        ret, frame = cap.read()
        if not ret:
            break
        inference_frames.append(frame)
        
        # Get model output
        if (cur_frame + 1) % 16 == 0 and cur_frame > 0:
            output = get_inference_output(inference_frames, model)
            if output == 0 and detected_text_delay <= 0:
                output_text = 'No flicker'
                output_color = green
            else:
                output_text = 'Flicker !!!'
                output_color = red
                if output != 0:
                    detected_text_delay = 32 # The line "Flicker !!!" will stay for at least 48 frames

        # Write frame with text line        
        add_text(frame, output_text, fps, color=output_color)
        detected_text_delay -= 1
        output_video_writer.write(frame)
        cur_frame += 1
    
    cap.release()
    output_video_writer.release()
    cv2.destroyAllWindows()