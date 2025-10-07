import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from joblib import load

class IntrusionDetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Intrusion Detection System")
        self.root.geometry("800x600")
        
        self.label = tk.Label(root, text="Intrusion Detection System", font=("Arial", 16, "bold"))
        self.label.pack(pady=10)
        
        self.load_button = tk.Button(root, text="Load Network Traffic File", command=self.load_file)
        self.load_button.pack(pady=5)
        
        self.predict_button = tk.Button(root, text="Start Detection", command=self.run_detection, state=tk.DISABLED)
        self.predict_button.pack(pady=5)
        
        self.text_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=100, height=20)
        self.text_area.pack(padx=10, pady=10)
        
        self.clear_button = tk.Button(root, text="Clear Logs", command=self.clear_logs)
        self.clear_button.pack(pady=5)
        
        self.quit_button = tk.Button(root, text="Exit", command=root.quit)
        self.quit_button.pack(pady=5)
        
        self.file_path = None
        self.model = load('random_forest_model1.joblib')
        self.scaler = load('scaler1.joblib')
        self.label_encoder = load('label_encoder1.joblib')
    
    def load_file(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if self.file_path:
            messagebox.showinfo("File Loaded", f"File loaded successfully: {self.file_path}")
            self.predict_button.config(state=tk.NORMAL)
    
    def run_detection(self):
        if not self.file_path:
            messagebox.showerror("Error", "No file selected!")
            return
        
        captured_data = pd.read_csv(self.file_path)
        top_features = ['Avg Bwd Segment Size', 'Total Length of Bwd Packets',
                        'Bwd Packet Length Max', 'Flow IAT Mean', 'Bwd Packet Length Mean',
                        'Flow Packets/s', 'Bwd Packets/s', 'Fwd IAT Total', 'Flow IAT Max', 'Destination Port']
        
        if not all(feature in captured_data.columns for feature in top_features):
            messagebox.showerror("Error", "Captured data does not have the required features!")
            return
        
        X_captured = captured_data[top_features].fillna(0)
        X_captured_scaled = self.scaler.transform(X_captured)
        predicted_labels = self.model.predict(X_captured_scaled)
        predicted_labels_mapped = self.label_encoder.inverse_transform(predicted_labels)
        captured_data['Predicted Label'] = predicted_labels_mapped
        captured_data.to_csv('captured_traffic_with_predictions.csv', index=False)
        
        self.text_area.insert(tk.END, "Detection Results:\n")
        for index, row in captured_data.iterrows():
            self.text_area.insert(tk.END, f"{row['Flow (SRC IP, DST IP, SRC Port, DST Port)']} - {row['Predicted Label']}\n")
            if row['Predicted Label'] != 'Benign':
                self.text_area.tag_add("alert", f"{index + 2}.0", f"{index + 2}.end")
                self.text_area.tag_config("alert", foreground="red")
    
    def clear_logs(self):
        self.text_area.delete('1.0', tk.END)

if __name__ == "__main__":
    root = tk.Tk()
    app = IntrusionDetectionGUI(root)
    root.mainloop()
