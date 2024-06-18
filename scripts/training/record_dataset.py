from data import CarDataset, ResizeWithLabels, RandomVerticalFlipWithLabel, RandomHorizontalFlipWithLabel, ChangeStreetColor, ComposeTransformations
import cv2
import numpy as np
from torch.utils.data import Subset

transforms = ComposeTransformations([
    ResizeWithLabels(),
    ChangeStreetColor()
])

car_dataset = CarDataset(root='../../data2', transform=transforms)
train_size = int(0.6 * len(car_dataset))
val_size = int(0.2 * len(car_dataset))
test_size = len(car_dataset) - train_size - val_size

test_indices = list(range(train_size + val_size, len(car_dataset)))
test_dataset = Subset(car_dataset, test_indices)

# Video writer.
frame_height, frame_width = car_dataset[0][0].size
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('brown_video.avi', fourcc, 20.0, (frame_width, frame_height))

# Display frames one by one, apply augmentations, and write to video.
for i in range(len(test_dataset)):
    frame, label = test_dataset[i]
    frame = np.array(frame)  # So the image can be displayed using OpenCv.

    # Convert RGB (used by PIL and Albumentations) to BGR (used by OpenCV)
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Display the frame
    cv2.imshow('Frame with Changed Street Colour', frame_bgr)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    # Write the frame to the video file
    out.write(frame_bgr)

# Release everything when done
out.release()
cv2.destroyAllWindows()

