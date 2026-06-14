import cv2
import os
import pandas as pd
from hand_tracking import HandTracker 

def collect_data():
    os.makedirs('../data/raw', exist_ok=True)
    label = input("Enter the label for this sign (e.g., A, B, Hello): ").strip().upper()
    csv_path = f'../data/raw/{label}.csv'
    
    cap = cv2.VideoCapture(0)
    tracker = HandTracker(max_hands=1)
    data = []
    
    print(f"Recording for '{label}'. Press 's' to save a frame. Press 'q' to quit.")
    
    while True:
        success, frame = cap.read()
        if not success: break
            
        frame = cv2.flip(frame, 1) 
        frame = tracker.find_hands(frame)
        landmarks = tracker.get_landmarks(frame)
        
        cv2.putText(frame, f"Label: {label} | Saved: {len(data)}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Data Collection", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            if len(landmarks) == 63: 
                data.append(landmarks)
                print(f"Saved frame! Total: {len(data)}")
        elif key == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    
    if data:
        columns = [f'{axis}{i}' for i in range(21) for axis in ('x', 'y', 'z')]
        df = pd.DataFrame(data, columns=columns)
        if os.path.exists(csv_path):
            df.to_csv(csv_path, mode='a', header=False, index=False)
        else:
            df.to_csv(csv_path, index=False)
        print(f"Saved {len(data)} rows to {csv_path}")

if __name__ == "__main__":
    collect_data()