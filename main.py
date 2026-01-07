
import tkinter as tk
from tkinter import messagebox, filedialog, scrolledtext, Toplevel, Checkbutton, IntVar
import pandas as pd
import numpy as np
import os
import cv2
import pickle
from PIL import Image, ImageTk
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
import sys
import threading
import time
from datetime import datetime
import pyperclip
import pyautogui
from scipy.spatial.distance import euclidean
from tkinter import simpledialog

# Determine if running as a PyInstaller bundle

if getattr(sys, 'frozen', False):
    base_path = sys._MEIPASS
else:
    base_path = os.path.dirname(__file__)

# Initialize FaceNet model and MTCNN
device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20, keep_all=True, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# File paths
csv_path = os.path.join(base_path, "dataset.csv")
excel_path = os.path.join(base_path, "attendance.xlsx")
encodings_path = os.path.join(base_path, "encodings.pkl")

# Global variables
video = None
is_camera_running = False
session_detected_students_times = {}
whatsapp_sent_for_day = set()
# Replace current_day with current_date (string)
current_date = datetime.now().strftime("%Y-%m-%d")
excel_display_visible = False
known_encodings = []
known_names = []
known_encodings_np = None
student_whatsapp = {}
RECOGNITION_THRESHOLD = 0.9
excel_lock = threading.Lock()

# UI elements (defined globally)
root = tk.Tk()
status_label = None
day_display_label = None
camera_label = None
excel_frame = None
excel_text = None
show_excel_button = None
button_frame = None

def format_time_am_pm():
    """Returns current time in 12-hour AM/PM format."""
    return datetime.now().strftime("%I:%M:%S %p")

def get_latest_date_from_excel():
    """Returns the latest attendance date column from Excel in YYYY-MM-DD format, or today's date if not found."""
    if os.path.exists(excel_path):
        try:
            df_attendance = pd.read_excel(excel_path)
            date_columns = [col for col in df_attendance.columns if _is_date_column(col)]
            if date_columns:
                # Return the latest date
                return max(date_columns)
        except Exception as e:
            print(f"Error reading Excel for latest date: {e}")
    return datetime.now().strftime("%Y-%m-%d")  # Default to today if not found

def _is_date_column(col):
    # Helper to check if a column is a date in YYYY-MM-DD format
    try:
        datetime.strptime(col, "%Y-%m-%d")
        return True
    except Exception:
        return False

def setup_initial_attendance_file():
    global current_date, known_encodings, known_names, known_encodings_np, student_whatsapp
    try:
        df_temp = pd.read_csv(csv_path)
        for _, row in df_temp.iterrows():
            number = str(row['S_parents_number']).strip()
            if number and number[0] != '+' and number.isdigit():
                number = f"+92{number}"
            if number.startswith('+') and number[1:].isdigit():
                student_whatsapp[f"{row['Student_ID']}_{row['Name'].replace(' ', '_')}"] = number
    except FileNotFoundError:
        messagebox.showerror("Error", f"dataset.csv not found at {csv_path}.")
        sys.exit()

    if os.path.exists(encodings_path):
        try:
            with open(encodings_path, "rb") as f:
                known_encodings, known_names = pickle.load(f)
            known_encodings_np = np.array(known_encodings)
            status_label.config(text=f"Loaded encodings for {len(known_encodings)} faces.")
        except Exception as e:
            messagebox.showwarning("Encodings Error", f"Could not load encodings.pkl: {e}.")
            status_label.config(text="❌ Error loading encodings.pkl")
            known_encodings = []
            known_names = []
            known_encodings_np = np.array([])
    else:
        messagebox.showwarning("Encodings Missing", f"'{encodings_path}' not found.")
        status_label.config(text="❌ Encodings missing. Run generate_encodings.py")

    today_str = datetime.now().strftime("%Y-%m-%d")
    if os.path.exists(excel_path):
        try:
            df_attendance = pd.read_excel(excel_path)
            if not all(col in df_attendance.columns for col in ["Student_ID", "Name"]):
                backup_path = os.path.join(base_path, f"attendance_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
                df_attendance.to_excel(backup_path, index=False)
                df_students_initial = pd.DataFrame({"Student_ID": df_temp['Student_ID'], "Name": df_temp['Name']}).drop_duplicates()
                df_attendance = pd.DataFrame({"Student_ID": df_students_initial['Student_ID'], "Name": df_students_initial['Name']})
                df_attendance[today_str] = pd.NA  # Ensure today's attendance date column exists
                df_attendance.to_excel(excel_path, index=False)
                current_date = today_str
            else:
                # Find the latest attendance date column
                date_columns = [col for col in df_attendance.columns if _is_date_column(col)]
                if not date_columns:
                    df_attendance[today_str] = pd.NA
                    df_attendance.to_excel(excel_path, index=False)
                    current_date = today_str
                else:
                    current_date = max(date_columns)
            update_day_display()
        except Exception as e:
            messagebox.showerror("Error", f"Error reading attendance.xlsx: {e}. Creating new.")
            df_students_initial = pd.DataFrame({"Student_ID": df_temp['Student_ID'], "Name": df_temp['Name']}).drop_duplicates()
            df_attendance = pd.DataFrame({"Student_ID": df_students_initial['Student_ID'], "Name": df_students_initial['Name']})
            df_attendance[today_str] = pd.NA  # Ensure today's attendance date column exists
            df_attendance.to_excel(excel_path, index=False)
            current_date = today_str
            update_day_display()
    else:
        df_students_initial = pd.DataFrame({"Student_ID": df_temp['Student_ID'], "Name": df_temp['Name']}).drop_duplicates()
        df_attendance = pd.DataFrame({"Student_ID": df_students_initial['Student_ID'], "Name": df_students_initial['Name']})
        df_attendance[today_str] = pd.NA  # Ensure today's attendance date column exists
        df_attendance.to_excel(excel_path, index=False)
        current_date = today_str
        update_day_display()

def update_day_display():
    """Updates the attendance date display label."""
    if day_display_label:
        root.after(0, lambda: day_display_label.config(text=f"Current Attendance Date: {current_date}"))

def _next_day_excel_operations():
    global current_date, whatsapp_sent_for_day
    with excel_lock:
        try:
            df_attendance = pd.read_excel(excel_path)
            previous_date_column = current_date
            if previous_date_column in df_attendance.columns:
                df_attendance[previous_date_column] = df_attendance[previous_date_column].astype(object).fillna("Absent")
            current_date = (datetime.strptime(current_date, "%Y-%m-%d") + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            new_date_column = current_date
            if new_date_column not in df_attendance.columns:
                df_attendance[new_date_column] = pd.NA
            df_attendance.to_excel(excel_path, index=False)
            root.after(0, lambda: status_label.config(text=f"Switched to {current_date}."))
            root.after(0, update_day_display)
            whatsapp_sent_for_day.clear()
            if excel_display_visible:
                root.after(0, display_excel)
        except Exception as e:
            root.after(0, lambda: messagebox.showerror("Error", f"Error managing Excel for next attendance date: {e}"))
            current_date = (datetime.strptime(current_date, "%Y-%m-%d") - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            root.after(0, update_day_display)

def handle_next_day():
    if messagebox.askyesno("Confirm Next Day", "Switch to next day? This will finalize current day's attendance."):
        if is_camera_running:
            messagebox.showwarning("Camera Active", "Stop the camera first.")
            return
        threading.Thread(target=_next_day_excel_operations, daemon=True).start()
        status_label.config(text="Processing next day...")

def record_single_attendance(student_id, name, detected_time_str):
    day_column = current_date
    full_name_key = f"{student_id}_{name.replace(' ', '_')}"
    with excel_lock:
        try:
            df_attendance = pd.read_excel(excel_path)
            df_attendance['Student_ID'] = df_attendance['Student_ID'].astype(str)
            df_attendance['Name'] = df_attendance['Name'].str.strip()
            if day_column not in df_attendance.columns:
                df_attendance[day_column] = pd.NA
            student_row_idx = df_attendance[(df_attendance['Student_ID'] == student_id) & (df_attendance['Name'] == name)].index
            if not student_row_idx.empty:
                idx = student_row_idx[0]
                if pd.isna(df_attendance.at[idx, day_column]) or "Absent" in str(df_attendance.at[idx, day_column]):
                    df_attendance.at[idx, day_column] = f"Present {detected_time_str}"
                    df_attendance.to_excel(excel_path, index=False)
                    if full_name_key not in whatsapp_sent_for_day:
                        number = student_whatsapp.get(full_name_key)
                        if number:
                            threading.Thread(target=send_whatsapp_message, args=(number, f"📢 Your child {name} entered at {detected_time_str}")).start()
                            whatsapp_sent_for_day.add(full_name_key)
                    if excel_display_visible:
                        root.after(10, display_excel)
                    root.after(10, lambda: status_label.config(text=f"Live: {name} detected. {current_date} updated."))
        except Exception as e:
            print(f"Error during live Excel update for {name}: {e}")

def send_whatsapp_message(number, message):
    try:
        os.system(f'start whatsapp://send?phone={number}')
        time.sleep(5)
        pyperclip.copy(message)
        pyautogui.hotkey('ctrl', 'v')
        time.sleep(0.5)
        pyautogui.press('enter')
        time.sleep(1)
    except Exception as e:
        print(f"WhatsApp automation failed for {number}: {e}")

def start_camera():
    global video, is_camera_running, session_detected_students_times
    if is_camera_running:
        status_label.config(text="⚠ Camera already running.")
        return
    if known_encodings_np is None or known_encodings_np.size == 0:
        messagebox.showwarning("No Encodings", "No face encodings loaded.")
        status_label.config(text="❌ No encodings loaded.")
        return
    status_label.config(text=f"Starting camera for {current_date}...")
    video = cv2.VideoCapture(0)
    if not video.isOpened():
        status_label.config(text="❌ Could not open camera.")
        return
    is_camera_running = True
    session_detected_students_times.clear()
    threading.Thread(target=camera_loop, daemon=True).start()
    status_label.config(text=f"�� Camera running for {current_date}.")

def camera_loop():
    global video, is_camera_running
    while is_camera_running:
        ret, frame = video.read()
        if not ret:
            root.after(0, lambda: status_label.config(text="❌ Failed to grab frame."))
            break
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        boxes, _ = mtcnn.detect(pil_image)
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = [int(b) for b in box]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                face_img_pil = pil_image.crop((x1, y1, x2, y2))
                if face_img_pil.width == 0 or face_img_pil.height == 0:
                    continue
                face_tensor = torch.tensor(np.array(face_img_pil.resize((160, 160)))).permute(2, 0, 1).float() / 255.0
                face_tensor = face_tensor.unsqueeze(0).to(device)
                current_face_encoding = resnet(face_tensor).detach().cpu().numpy().flatten()
                name_display = "Unknown"
                full_name_key = "Unknown"
                if current_face_encoding is not None and known_encodings_np.size > 0:
                    distances = [euclidean(current_face_encoding, enc) for enc in known_encodings_np]
                    min_distance_idx = np.argmin(distances)
                    if distances[min_distance_idx] < RECOGNITION_THRESHOLD:
                        full_name_key = known_names[min_distance_idx]
                        student_id, original_name = full_name_key.split("_", 1)
                        name_display = original_name.replace("_", " ")
                        time_now = format_time_am_pm()
                        if full_name_key not in session_detected_students_times:
                            session_detected_students_times[full_name_key] = time_now
                            threading.Thread(target=record_single_attendance,
                                            args=(student_id, name_display, time_now)).start()
                color = (0, 255, 0) if name_display != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, name_display, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img.thumbnail((400, 300), Image.Resampling.LANCZOS)
        imgtk = ImageTk.PhotoImage(image=img)
        root.after(0, lambda: camera_label.config(image=imgtk))
        root.after(0, lambda: setattr(camera_label, 'imgtk', imgtk))
        time.sleep(0.01)
    if video:
        video.release()
        video = None
    root.after(0, lambda: status_label.config(text="🛑 Camera stopped."))
    root.after(0, lambda: camera_label.config(image=''))
    root.after(0, lambda: setattr(camera_label, 'image', None))

def stop_camera():
    global is_camera_running, video
    if not is_camera_running:
        status_label.config(text="⚠ Camera is not running.")
        return
    is_camera_running = False
    status_label.config(text="Stopping camera...")

def display_excel():
    with excel_lock:
        try:
            df_attendance = pd.read_excel(excel_path)
            root.after(0, lambda: excel_text.delete(1.0, tk.END))
            root.after(0, lambda: excel_text.insert(tk.END, df_attendance.to_string(index=False)))
        except FileNotFoundError:
            root.after(0, lambda: excel_text.delete(1.0, tk.END))
            root.after(0, lambda: excel_text.insert(tk.END, "Attendance file not found."))
        except Exception as e:
            root.after(0, lambda: messagebox.showerror("Error", f"Error displaying Excel: {e}"))

def toggle_excel_display():
    global excel_display_visible
    if excel_display_visible:
        excel_frame.pack_forget()
        excel_display_visible = False
        show_excel_button.config(text="📊 Show Attendance List")
    else:
        excel_frame.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
        display_excel()
        excel_display_visible = True
        show_excel_button.config(text="📊 Hide Attendance List")

def prompt_day_selection_and_download():
    with excel_lock:
        try:
            df_attendance = pd.read_excel(excel_path)
            all_date_columns = sorted([col for col in df_attendance.columns if _is_date_column(col)],
                                     key=lambda x: datetime.strptime(x, "%Y-%m-%d"))
            if not all_date_columns:
                messagebox.showinfo("No Dates", "No attendance dates to download.")
                return
            selection_window = Toplevel(root)
            selection_window.title("Select Attendance Dates to Download")
            selection_window.transient(root)
            selection_window.geometry("300x400")
            check_vars = []
            select_all_var = IntVar(value=0)
            def toggle_all_checkboxes():
                state = select_all_var.get()
                for var in check_vars:
                    var[1].set(state)
            tk.Checkbutton(selection_window, text="Select All", variable=select_all_var,
                          command=toggle_all_checkboxes, font=("Arial", 10, "bold")).pack(anchor='w', padx=10, pady=5)
            for date_col in all_date_columns:
                var = IntVar(value=1)
                chk = tk.Checkbutton(selection_window, text=date_col, variable=var)
                chk.pack(anchor='w', padx=20)
                check_vars.append((date_col, var))
            def perform_download():
                selected_dates = ["Student_ID", "Name"] + [date_col for date_col, var in check_vars if var.get() == 1]
                if len(selected_dates) <= 2:
                    messagebox.showwarning("No Dates Selected", "Select at least one attendance date.")
                    return
                df_to_download = df_attendance[selected_dates]
                download_path = filedialog.asksaveasfilename(
                    defaultextension=".xlsx",
                    filetypes=[("Excel files", "*.xlsx")],
                    title="Save Attendance Data"
                )
                if download_path:
                    try:
                        df_to_download.to_excel(download_path, index=False)
                        root.after(0, lambda: messagebox.showinfo("Success", f"Attendance downloaded to {download_path}"))
                    except Exception as e:
                        root.after(0, lambda: messagebox.showerror("Error", f"Error saving Excel: {e}"))
                selection_window.destroy()
            button_frame_selection = tk.Frame(selection_window)
            button_frame_selection.pack(pady=10)
            tk.Button(button_frame_selection, text="Download Selected", command=perform_download, bg="#4CAF50", fg="white", padx=10, pady=5).pack(side=tk.LEFT, padx=5)
            tk.Button(button_frame_selection, text="Cancel", command=selection_window.destroy, bg="#F44336", fg="white", padx=10, pady=5).pack(side=tk.LEFT, padx=5)
        except FileNotFoundError:
            messagebox.showerror("Error", "Attendance file not found.")
        except Exception as e:
            messagebox.showerror("Error", f"Error preparing download: {e}")

def custom_reset_dialog():
    dialog = Toplevel(root)
    dialog.title("Reset Previous Day Options")
    dialog.geometry("400x180")
    dialog.transient(root)
    dialog.grab_set()
    label = tk.Label(dialog, text="After deleting the current day, how should the previous day's attendance be updated?", wraplength=380, justify="left", font=("Arial", 11))
    label.pack(pady=10, padx=10)
    result = {'choice': None}
    def choose_reset():
        result['choice'] = 'reset'
        dialog.destroy()
    def choose_absent():
        result['choice'] = 'absent'
        dialog.destroy()
    btn_frame = tk.Frame(dialog)
    btn_frame.pack(pady=10)
    tk.Button(btn_frame, text="Reset All (set all to Absent)", command=choose_reset, bg="#E53935", fg="white", padx=10, pady=5).pack(side=tk.LEFT, padx=10)
    tk.Button(btn_frame, text="Only Reset Absentees (keep Present)", command=choose_absent, bg="#4CAF50", fg="white", padx=10, pady=5).pack(side=tk.LEFT, padx=10)
    dialog.wait_window()
    return result['choice']

def custom_reset_current_day_dialog():
    dialog = Toplevel(root)
    dialog.title("Reset Current Day Options")
    dialog.geometry("400x180")
    dialog.transient(root)
    dialog.grab_set()
    label = tk.Label(dialog, text="How should the current day's attendance be reset?", wraplength=380, justify="left", font=("Arial", 11))
    label.pack(pady=10, padx=10)
    result = {'choice': None}
    def choose_reset():
        result['choice'] = 'reset'
        dialog.destroy()
    def choose_absent():
        result['choice'] = 'absent'
        dialog.destroy()
    btn_frame = tk.Frame(dialog)
    btn_frame.pack(pady=10)
    tk.Button(btn_frame, text="Reset All (set all to NaN)", command=choose_reset, bg="#E53935", fg="white", padx=10, pady=5).pack(side=tk.LEFT, padx=10)
    tk.Button(btn_frame, text="Only Reset Absentees (keep Present)", command=choose_absent, bg="#4CAF50", fg="white", padx=10, pady=5).pack(side=tk.LEFT, padx=10)
    dialog.wait_window()
    return result['choice']

def reset_current_day_column():
    global current_date
    with excel_lock:
        try:
            df_attendance = pd.read_excel(excel_path)
            current_col = current_date
            if current_col not in df_attendance.columns:
                messagebox.showwarning("Not Found", f"{current_col} not found in Excel.")
                return
            user_choice = custom_reset_current_day_dialog()
            if user_choice == 'reset':
                df_attendance[current_col] = pd.NA
            else:
                df_attendance[current_col] = df_attendance[current_col].apply(lambda x: x if pd.notna(x) and str(x).startswith('Present') else pd.NA)
            df_attendance.to_excel(excel_path, index=False)
            if excel_display_visible:
                root.after(0, display_excel)
            root.after(0, lambda: status_label.config(text=f"Reset {current_col} attendance date as per user choice."))
        except Exception as e:
            root.after(0, lambda: messagebox.showerror("Error", f"Error resetting attendance date column: {e}"))

def delete_current_day_column():
    global current_date
    with excel_lock:
        try:
            df_attendance = pd.read_excel(excel_path)
            date_columns = [col for col in df_attendance.columns if _is_date_column(col)]
            if not date_columns:
                messagebox.showinfo("No Days", "No attendance day columns to delete.")
                return
            current_col = current_date
            # Prevent deleting the earliest date column
            earliest_date = min(date_columns, key=lambda x: datetime.strptime(x, "%Y-%m-%d"))
            if current_col == earliest_date:
                messagebox.showinfo("Protected", f"The first attendance day ({earliest_date}) cannot be deleted.")
                return
            if current_col not in df_attendance.columns:
                messagebox.showwarning("Not Found", f"{current_col} not found in Excel.")
                return
            # Remove the current day column
            df_attendance = df_attendance.drop(columns=[current_col])
            # Move back to previous day
            prev_dates = [datetime.strptime(col, "%Y-%m-%d") for col in date_columns if datetime.strptime(col, "%Y-%m-%d") < datetime.strptime(current_col, "%Y-%m-%d")]
            if prev_dates:
                current_date = max(prev_dates).strftime("%Y-%m-%d")
            else:
                # If no date columns left, create today's date
                date_cols_left = [col for col in df_attendance.columns if _is_date_column(col)]
                if not date_cols_left:
                    df_attendance[datetime.now().strftime("%Y-%m-%d")] = pd.NA
                    current_date = datetime.now().strftime("%Y-%m-%d")
                else:
                    current_date = datetime.now().strftime("%Y-%m-%d")
            df_attendance.to_excel(excel_path, index=False)
            update_day_display()
            if excel_display_visible:
                root.after(0, display_excel)
            root.after(0, lambda: status_label.config(text=f"Deleted {current_col}. Now at {current_date}."))
        except Exception as e:
            root.after(0, lambda: messagebox.showerror("Error", f"Error deleting day column: {e}"))

# UI Setup
root.title("Attendance System")
root.geometry("600x700")
root.configure(bg="#f0f0f0")

tk.Label(root, text="🏫 Attendance System", font=("Arial", 20, "bold"), bg="#4CAF50", fg="white", padx=10, pady=5).pack(fill=tk.X)
day_display_label = tk.Label(root, text=f"Current Attendance Date: {current_date}", font=("Arial", 14), bg="#f0f0f0")
day_display_label.pack(pady=5)
status_label = tk.Label(root, text="Ready", font=("Arial", 12), bg="#f0f0f0")
status_label.pack(pady=5, fill=tk.X)
camera_label = tk.Label(root, bg="#ffffff", borderwidth=2, relief="solid")
camera_label.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
button_frame = tk.Frame(root, bg="#f0f0f0")
button_frame.pack(pady=10, padx=10, fill=tk.X)
tk.Button(button_frame, text="🗓️ Next Day", command=handle_next_day, bg="#2196F3", fg="white", padx=10, pady=5, relief="raised", bd=3).pack(side=tk.LEFT, padx=5)
tk.Button(button_frame, text="📸 Start Camera", command=start_camera, bg="#FF9800", fg="white", padx=10, pady=5, relief="raised", bd=3).pack(side=tk.LEFT, padx=5)
tk.Button(button_frame, text="🛑 Stop Camera", command=stop_camera, bg="#F44336", fg="white", padx=10, pady=5, relief="raised", bd=3).pack(side=tk.LEFT, padx=5)
tk.Button(button_frame, text="⬇ Download Excel", command=prompt_day_selection_and_download, bg="#9C27B0", fg="white", padx=10, pady=5, relief="raised", bd=3).pack(side=tk.LEFT, padx=5)
tk.Button(button_frame, text="🧹 Reset Current Day", command=reset_current_day_column, bg="#FFC107", fg="black", padx=10, pady=5, relief="raised", bd=3).pack(side=tk.LEFT, padx=5)
tk.Button(button_frame, text="🗑️ Delete Current Day", command=delete_current_day_column, bg="#E53935", fg="white", padx=10, pady=5, relief="raised", bd=3).pack(side=tk.LEFT, padx=5)
excel_frame = tk.Frame(root, bg="#f0f0f0")
excel_text = scrolledtext.ScrolledText(excel_frame, height=10, width=70, bg="#ffffff", borderwidth=2, relief="solid")
excel_text.pack(pady=5, fill=tk.BOTH, expand=True)
show_excel_button = tk.Button(root, text="📊 Show Attendance List", command=toggle_excel_display, bg="#607D8B", fg="white", padx=10, pady=5, relief="raised", bd=3)
show_excel_button.pack(pady=5)

setup_initial_attendance_file()
root.mainloop()