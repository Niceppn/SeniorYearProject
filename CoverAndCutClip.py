import os
import subprocess
import threading
import tkinter as tk
from tkinter import messagebox, simpledialog
from tkinter.scrolledtext import ScrolledText

from openai import OpenAI
from pydub import AudioSegment

# เพิ่ม import สำหรับ Whisper Local
try:
    import whisper
    WHISPER_LOCAL_AVAILABLE = True
except ImportError:
    WHISPER_LOCAL_AVAILABLE = False




# Whisper Local Model (ถ้ามี)
whisper_model = None

OUTPUT_FOLDER = "downloads"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def download_audio(url, callback):
    """ดาวน์โหลดเสียง mp3 จาก yt-dlp (รองรับ YouTube, TikTok และอื่นๆ)"""
    try:
        # สร้างชื่อไฟล์ที่ปลอดภัยสำหรับ filesystem
        output_template = os.path.join(OUTPUT_FOLDER, "%(title).100s.%(ext)s")
        command = [
            "yt-dlp",
            "-x",
            "--audio-format", "mp3",
            "-o", output_template,
            url,
            "--no-warnings",  
            "--restrict-filenames" 
        ]
        
        print(f"กำลังรันคำสั่ง: {' '.join(command)}")
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"ผลลัพธ์: {result.stdout}")
        
        # หาไฟล์ mp3 ล่าสุดในโฟลเดอร์
        files = [f for f in os.listdir(OUTPUT_FOLDER) if f.endswith(".mp3")]
        if not files:
            raise Exception("ไม่พบไฟล์เสียงที่ดาวน์โหลด")
        latest_file = max(files, key=lambda f: os.path.getctime(os.path.join(OUTPUT_FOLDER, f)))
        callback(os.path.join(OUTPUT_FOLDER, latest_file))
    except subprocess.CalledProcessError as e:
        error_msg = f"ไม่สามารถดาวน์โหลดได้\nReturn code: {e.returncode}"
        if hasattr(e, 'stderr') and e.stderr:
            error_msg += f"\nError: {e.stderr}"
        messagebox.showerror("Download Error", f"{error_msg}\n\nตรวจสอบ:\n- ลิงก์ถูกต้องหรือไม่\n- เชื่อมต่ออินเทอร์เน็ตหรือไม่\n- วิดีโอเป็น Private หรือไม่")
    except Exception as e:
        messagebox.showerror("Download Error", f"เกิดข้อผิดพลาด: {e}")

def transcribe_with_openai(audio_path):
    """ถอดเสียงด้วย OpenAI Whisper API"""
    try:
        with open(audio_path, "rb") as audio_file:
            # ใช้ OpenAI Whisper API เวอร์ชันใหม่
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="th",  # ระบุภาษาไทย
                response_format="verbose_json",  # ให้ส่ง timestamps กลับมา
                timestamp_granularities=["segment"]
            )
        return transcript
    except Exception as e:
        messagebox.showerror("Transcription Error", f"ไม่สามารถถอดเสียงได้\n{e}")
        return None

def transcribe_with_whisper_local(audio_path, model_size="large-v3"):
    """ถอดเสียงด้วย Whisper Local (ฟรี แต่ต้องการ GPU)"""
    global whisper_model
    try:
        # โหลด model ถ้ายังไม่ได้โหลด
        if whisper_model is None:
            messagebox.showinfo("Loading Model", f"กำลังโหลด Whisper {model_size} model... กรุณารอสักครู่")
            whisper_model = whisper.load_model(model_size)
        
        # ถอดเสียง
        result = whisper_model.transcribe(
            audio_path, 
            language="th",
            word_timestamps=True,
            verbose=True
        )
        
        # แปลงเป็นรูปแบบเดียวกับ OpenAI
        segments = []
        for i, segment in enumerate(result.get('segments', [])):
            segments.append({
                'start': segment.get('start', 0),
                'end': segment.get('end', 0),
                'text': segment.get('text', '').strip()
            })
        
        return {'segments': segments}
    except Exception as e:
        messagebox.showerror("Local Whisper Error", f"ไม่สามารถถอดเสียงด้วย Whisper Local ได้\n{e}")
        return None

def transcribe_with_enhanced_openai(audio_path):
    """ถอดเสียงด้วย OpenAI แบบปรับแต่งให้แม่นยำขึ้น"""
    try:
        with open(audio_path, "rb") as audio_file:
            # ใช้การตั้งค่าที่ปรับแต่งเพื่อความแม่นยำสูงสุด
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="th",
                response_format="verbose_json",
                timestamp_granularities=["segment"],
                # เพิ่มการตั้งค่าสำหรับความแม่นยำ
                prompt="นี่คือการสนทนาภาษาไทย กรุณาถอดเสียงให้แม่นยำที่สุด",  # ช่วยให้ AI เข้าใจบริบท
                temperature=0.0  # ลดความสุ่มเพื่อความแม่นยำ
            )
        return transcript
    except Exception as e:
        messagebox.showerror("Enhanced Transcription Error", f"ไม่สามารถถอดเสียงได้\n{e}")
        return None

def transcribe_and_show(audio_path, text_widget, segments_listbox, method="openai"):
    """ถอดเสียงและแสดง segment"""
    if method == "openai":
        result = transcribe_with_openai(audio_path)
    elif method == "enhanced":
        result = transcribe_with_enhanced_openai(audio_path)
    elif method == "local" and WHISPER_LOCAL_AVAILABLE:
        result = transcribe_with_whisper_local(audio_path)
    else:
        messagebox.showerror("Error", "วิธีการถอดเสียงที่เลือกไม่พร้อมใช้งาน")
        return []
    
    if not result:
        return []
    
    # แปลง response object เป็น dictionary หากจำเป็น
    if hasattr(result, 'model_dump'):
        result_dict = result.model_dump()
    elif hasattr(result, 'dict'):
        result_dict = result.dict()
    else:
        result_dict = result
    
    segments = result_dict.get('segments', [])
    
    # แสดงข้อความลงใน widget
    text_widget.config(state=tk.NORMAL)
    text_widget.delete("1.0", tk.END)
    segments_listbox.delete(0, tk.END)

    for i, seg in enumerate(segments):
        start = seg.get('start', 0)
        end = seg.get('end', 0)
        text = seg.get('text', '').strip()
        text_widget.insert(tk.END, f"[{i}] ({start:.2f}s - {end:.2f}s): {text}\n")
        segments_listbox.insert(tk.END, f"{i}: {text}")

    text_widget.config(state=tk.DISABLED)
    return segments

def cut_audio_segment(audio_path, segment, output_name, delete_original=False):
    """ตัดเสียงจาก segment และบันทึกไฟล์ใหม่"""
    try:
        start_ms = int(segment.get('start', 0) * 1000)
        end_ms = int(segment.get('end', 0) * 1000)

        audio = AudioSegment.from_file(audio_path)
        clip = audio[start_ms:end_ms]

        clip.export(output_name, format="mp3")
        
        success_msg = f"บันทึกไฟล์เสียงตัดต่อ: {output_name}"
        
        # ลบไฟล์ต้นฉบับถ้าต้องการ
        if delete_original:
            try:
                os.remove(audio_path)
                success_msg += f"\n🗑️ ลบไฟล์ต้นฉบับแล้ว: {os.path.basename(audio_path)}"
            except Exception as e:
                success_msg += f"\n⚠️ ไม่สามารถลบไฟล์ต้นฉบับได้: {e}"
        
        messagebox.showinfo("Success", success_msg)
        
    except Exception as e:
        messagebox.showerror("Error", f"ไม่สามารถตัดเสียงได้: {e}")

def validate_url(url):
    """ตรวจสอบและระบุประเภทของ URL"""
    url = url.strip().lower()
    
    # รายการ domains ที่รองรับ
    supported_platforms = {
        'youtube.com': 'YouTube',
        'youtu.be': 'YouTube',
        'm.youtube.com': 'YouTube',
        'tiktok.com': 'TikTok',
        'www.tiktok.com': 'TikTok',
        'vm.tiktok.com': 'TikTok',
        'vt.tiktok.com': 'TikTok',
        'm.tiktok.com': 'TikTok'
    }
    
    for domain, platform in supported_platforms.items():
        if domain in url:
            return True, platform
    
    # ถ้าไม่ตรงกับ platform ที่รู้จัก แต่เป็น URL ให้ลองดู
    if url.startswith(('http://', 'https://')):
        return True, 'อื่นๆ'
    
    return False, None

def get_platform_emoji(platform):
    """ส่งคืน emoji ตาม platform"""
    emojis = {
        'YouTube': '🔴',
        'TikTok': '⚫',
        'อื่นๆ': '🌐'
    }
    return emojis.get(platform, '🌐')

def parse_multiple_urls(text):
    """แยกหลาย URL จากข้อความ"""
    import re
    
    # ค้นหา URLs ในข้อความ
    url_pattern = r'https?://[^\s\n]+'
    urls = re.findall(url_pattern, text)
    
    # ถ้าไม่เจอ URLs แบบ regex ให้แยกตามบรรทัด
    if not urls:
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        urls = [line for line in lines if line.startswith(('http://', 'https://'))]
    
    return urls

def download_multiple_audio(urls, progress_callback, completion_callback):
    """ดาวน์โหลดเสียงจากหลาย URL"""
    downloaded_files = []
    total_urls = len(urls)
    
    for i, url in enumerate(urls):
        try:
            # อัปเดตสถานะ
            is_valid, platform = validate_url(url)
            if not is_valid:
                progress_callback(i + 1, total_urls, f"❌ ลิงก์ไม่ถูกต้อง: {url[:50]}...")
                continue
            
            platform_emoji = get_platform_emoji(platform)
            progress_callback(i + 1, total_urls, f"⏳ {platform_emoji} กำลังดาวน์โหลดจาก {platform}...")
            
            # ดาวน์โหลด
            output_template = os.path.join(OUTPUT_FOLDER, f"%(title).100s_clip{i+1}.%(ext)s")
            command = [
                "yt-dlp",
                "-x",
                "--audio-format", "mp3",
                "-o", output_template,
                url,
                "--no-warnings",
                "--restrict-filenames"
            ]
            
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            
            # หาไฟล์ที่ดาวน์โหลดล่าสุด
            files_before = set(f for f in os.listdir(OUTPUT_FOLDER) if f.endswith(".mp3"))
            files_after = set(f for f in os.listdir(OUTPUT_FOLDER) if f.endswith(".mp3"))
            new_files = files_after - files_before
            
            if not new_files:
                # ถ้าไม่มีไฟล์ใหม่ ให้หาไฟล์ล่าสุดที่มี clip{i+1}
                all_files = [f for f in os.listdir(OUTPUT_FOLDER) if f.endswith(".mp3") and f"clip{i+1}" in f]
                if all_files:
                    latest_file = max(all_files, key=lambda f: os.path.getctime(os.path.join(OUTPUT_FOLDER, f)))
                    downloaded_files.append(os.path.join(OUTPUT_FOLDER, latest_file))
                else:
                    # หาไฟล์ล่าสุดทั่วไป
                    all_files = [f for f in os.listdir(OUTPUT_FOLDER) if f.endswith(".mp3")]
                    if all_files:
                        latest_file = max(all_files, key=lambda f: os.path.getctime(os.path.join(OUTPUT_FOLDER, f)))
                        downloaded_files.append(os.path.join(OUTPUT_FOLDER, latest_file))
            else:
                latest_file = list(new_files)[0]
                downloaded_files.append(os.path.join(OUTPUT_FOLDER, latest_file))
            
            progress_callback(i + 1, total_urls, f"✅ {platform_emoji} ดาวน์โหลดเสร็จ: {platform}")
            
        except subprocess.CalledProcessError as e:
            progress_callback(i + 1, total_urls, f"❌ ดาวน์โหลดไม่สำเร็จ: {url[:50]}...")
            continue
        except Exception as e:
            progress_callback(i + 1, total_urls, f"❌ เกิดข้อผิดพลาด: {str(e)[:50]}...")
            continue
    
    completion_callback(downloaded_files)

def process_multiple_files(audio_files, method, progress_callback, completion_callback):
    """ถอดเสียงจากหลายไฟล์"""
    all_results = []
    total_files = len(audio_files)
    
    for i, audio_path in enumerate(audio_files):
        try:
            progress_callback(i + 1, total_files, f"⏳ กำลังถอดเสียงไฟล์ {i+1}/{total_files}...")
            
            # ถอดเสียง
            if method == "openai":
                result = transcribe_with_openai(audio_path)
            elif method == "enhanced":
                result = transcribe_with_enhanced_openai(audio_path)
            elif method == "local" and WHISPER_LOCAL_AVAILABLE:
                result = transcribe_with_whisper_local(audio_path)
            else:
                continue
            
            if result:
                # แปลง response object เป็น dictionary หากจำเป็น
                if hasattr(result, 'model_dump'):
                    result_dict = result.model_dump()
                elif hasattr(result, 'dict'):
                    result_dict = result.dict()
                else:
                    result_dict = result
                
                segments = result_dict.get('segments', [])
                all_results.append({
                    'file': os.path.basename(audio_path),
                    'path': audio_path,
                    'segments': segments
                })
            
            progress_callback(i + 1, total_files, f"✅ ถอดเสียงเสร็จ ไฟล์ {i+1}/{total_files}")
            
        except Exception as e:
            progress_callback(i + 1, total_files, f"❌ ถอดเสียงไม่สำเร็จ ไฟล์ {i+1}")
            continue
    
    completion_callback(all_results)

def setup_api_key():
    """ตั้งค่า API Key ผ่าน dialog"""
    global client
    try:
        # ตรวจสอบว่า API key ยังไม่ได้ตั้งค่าหรือเป็นค่าเริ่มต้น
        if not hasattr(client, 'api_key') or client.api_key == "your-api-key-here" or not client.api_key:
            api_key = simpledialog.askstring(
                show='*'
            )
            if api_key:
                client = OpenAI(api_key=api_key)
                messagebox.showinfo("Success", "ตั้งค่า API Key เรียบร้อย!")
                return True
            else:
                messagebox.showerror("Error", "ต้องใส่ API Key ก่อนใช้งาน")
                return False
        return True
    except Exception as e:
        api_key = simpledialog.askstring(
            "OpenAI API Key", 
            "กรุณาใส่ OpenAI API Key:\n(หาได้จาก https://platform.openai.com/api-keys)",
            show='*'
        )
        if api_key:
            client = OpenAI(api_key=api_key)
            messagebox.showinfo("Success", "ตั้งค่า API Key เรียบร้อย!")
            return True
        else:
            messagebox.showerror("Error", "ต้องใส่ API Key ก่อนใช้งาน")
            return False

def on_download_click():
    # ตรวจสอบ URL ก่อน
    url_text = url_entry.get("1.0", tk.END).strip()
    if not url_text:
        messagebox.showerror("Error", "กรุณาใส่ลิงก์ก่อน")
        return
    
    # แยกหลาย URLs
    urls = parse_multiple_urls(url_text)
    if not urls:
        messagebox.showerror("Error", "ไม่พบลิงก์ที่ถูกต้อง\nรองรับ: YouTube, TikTok และอื่นๆ")
        return
    
    # ตรวจสอบจำนวน URLs
    if len(urls) > 20:
        response = messagebox.askyesno(
            "จำนวนลิงก์เยอะ", 
            f"พบ {len(urls)} ลิงก์ ซึ่งอาจใช้เวลานาน\nต้องการดำเนินการต่อหรือไม่?"
        )
        if not response:
            return
    
    # แสดงข้อมูลที่ตรวจพบ
    status_label.config(text=f"🔍 พบ {len(urls)} ลิงก์")
    window.update()
    
    # ตรวจสอบ API Key สำหรับ OpenAI methods
    selected_method = transcription_method.get()
    if selected_method in ["openai", "enhanced"] and not setup_api_key():
        return

    # ตรวจสอบว่ามี Whisper Local หรือไม่
    if selected_method == "local" and not WHISPER_LOCAL_AVAILABLE:
        response = messagebox.askyesno(
            "Whisper Local ไม่พร้อมใช้งาน",
            "ต้องติดตั้ง openai-whisper ก่อน\nต้องการติดตั้งตอนนี้หรือไม่?"
        )
        if response:
            install_whisper_local()
        return

    method_text = {
        "openai": "OpenAI Whisper API",
        "enhanced": "OpenAI Whisper (Enhanced)",
        "local": "Whisper Local (Large-v3)"
    }

    # เริ่มการประมวลผล
    def download_progress(current, total, message):
        status_label.config(text=f"📥 [{current}/{total}] {message}")
        window.update()
    
    def download_complete(audio_files):
        if not audio_files:
            status_label.config(text="❌ ไม่สามารถดาวน์โหลดไฟล์ใดได้")
            return
        
        status_label.config(text=f"✅ ดาวน์โหลดเสร็จ {len(audio_files)} ไฟล์")
        window.update()
        
        # เริ่มถอดเสียง
        def transcribe_progress(current, total, message):
            status_label.config(text=f"🎤 [{current}/{total}] {message}")
            window.update()
        
        def transcribe_complete(results):
            if not results:
                status_label.config(text="❌ ไม่สามารถถอดเสียงไฟล์ใดได้")
                return
            
            # แสดงผลลัพธ์ทั้งหมด
            display_multiple_results(results)
            status_label.config(text=f"✅ เสร็จสิ้น! ถอดเสียงได้ {len(results)} ไฟล์")
            
            global all_results_global
            all_results_global = results
        
        # เริ่มถอดเสียงในเธรดแยก
        def transcribe_task():
            process_multiple_files(audio_files, selected_method, transcribe_progress, transcribe_complete)
        
        threading.Thread(target=transcribe_task).start()
    
    # เริ่มดาวน์โหลดในเธรดแยก
    def download_task():
        download_multiple_audio(urls, download_progress, download_complete)
    
    threading.Thread(target=download_task).start()

def display_multiple_results(results):
    """แสดงผลลัพธ์จากหลายไฟล์"""
    # ล้างข้อมูลเก่า
    transcript_text.config(state=tk.NORMAL)
    transcript_text.delete("1.0", tk.END)
    segments_listbox.delete(0, tk.END)
    
    # แสดงผลลัพธ์แต่ละไฟล์
    global segments_global
    segments_global = []
    
    for i, result in enumerate(results):
        filename = result['file']
        segments = result['segments']
        
        # เพิ่มหัวข้อไฟล์
        transcript_text.insert(tk.END, f"\n📁 ไฟล์ {i+1}: {filename}\n")
        transcript_text.insert(tk.END, "=" * 80 + "\n")
        
        # เพิ่ม segments
        for j, seg in enumerate(segments):
            start = seg.get('start', 0)
            end = seg.get('end', 0)
            text = seg.get('text', '').strip()
            
            segment_index = len(segments_global)
            segment_with_file = {
                'start': start,
                'end': end,
                'text': text,
                'file_index': i,
                'file_path': result['path'],
                'filename': filename
            }
            segments_global.append(segment_with_file)
            
            transcript_text.insert(tk.END, f"[{segment_index}] ({start:.2f}s - {end:.2f}s): {text}\n")
            segments_listbox.insert(tk.END, f"{segment_index}: [{filename}] {text}")
    
    transcript_text.config(state=tk.DISABLED)

def on_cut_click():
    """ตัดเสียงจาก segment ที่เลือก (รองรับหลายไฟล์)"""
    selected = segments_listbox.curselection()
    if not selected:
        messagebox.showwarning("Warning", "กรุณาเลือก segment ก่อน")
        return

    index = selected[0]
    
    # ตรวจสอบว่าใช้ระบบเก่าหรือใหม่
    if len(segments_global) > 0 and isinstance(segments_global[index], dict) and 'file_path' in segments_global[index]:
        # ระบบใหม่ (multiple files)
        segment = segments_global[index]
        audio_path = segment['file_path']
        filename_prefix = f"cut_{segment['filename'].replace('.mp3', '')}"
        original_filename = segment['filename']
    else:
        # ระบบเก่า (single file)
        segment = segments_global[index]
        audio_path = current_audio_path
        filename_prefix = "cut"
        original_filename = os.path.basename(audio_path) if audio_path else "unknown"
    
    # Dialog สำหรับตั้งชื่อไฟล์
    filename = simpledialog.askstring(
        "ตั้งชื่อไฟล์", 
        f"ชื่อไฟล์ mp3 ที่จะบันทึก (ไม่ต้องใส่นามสกุล):\nแนะนำ: {filename_prefix}_segment{index}"
    )
    if not filename:
        return

    # ถามว่าต้องการลบไฟล์ต้นฉบับหรือไม่
    delete_original = messagebox.askyesno(
        "ลบไฟล์ต้นฉบับ?", 
        f"ต้องการลบไฟล์ต้นฉบับหลังจากตัดเสร็จหรือไม่?\n\n"
        f"ไฟล์ต้นฉบับ: {original_filename}\n"
        f"ไฟล์ใหม่: {filename}.mp3\n\n"
        f"⚠️ หากเลือก 'Yes' ไฟล์ต้นฉบับจะถูกลบถาวร!"
    )

    output_path = os.path.join(OUTPUT_FOLDER, f"{filename}.mp3")
    cut_audio_segment(audio_path, segment, output_path, delete_original)

def install_whisper_local():
    """ติดตั้ง Whisper Local"""
    def install_task():
        try:
            status_label.config(text="⏳ กำลังติดตั้ง openai-whisper...")
            window.update()
            subprocess.run(["pip", "install", "openai-whisper"], check=True)
            messagebox.showinfo("Success", "ติดตั้ง openai-whisper เรียบร้อย! กรุณารีสตาร์ทโปรแกรม")
        except subprocess.CalledProcessError as e:
            messagebox.showerror("Install Error", f"ไม่สามารถติดตั้งได้\n{e}")
        finally:
            status_label.config(text="")
            window.update()
    
    threading.Thread(target=install_task).start()

# เพิ่มฟังก์ชันจัดการไฟล์
def cleanup_original_files():
    """ลบไฟล์ต้นฉบับทั้งหมดที่ยังเหลืออยู่"""
    try:
        files = [f for f in os.listdir(OUTPUT_FOLDER) if f.endswith(".mp3") and not f.startswith("cut_")]
        if not files:
            messagebox.showinfo("Cleanup", "ไม่มีไฟล์ต้นฉบับให้ลบ")
            return
        
        response = messagebox.askyesno(
            "ลบไฟล์ต้นฉบับทั้งหมด", 
            f"พบไฟล์ต้นฉบับ {len(files)} ไฟล์\nต้องการลบทั้งหมดหรือไม่?\n\n⚠️ การกระทำนี้ไม่สามารถย้อนกลับได้!"
        )
        
        if response:
            deleted_count = 0
            for file in files:
                try:
                    os.remove(os.path.join(OUTPUT_FOLDER, file))
                    deleted_count += 1
                except Exception as e:
                    print(f"ไม่สามารถลบ {file}: {e}")
            
            messagebox.showinfo("Cleanup Complete", f"ลบไฟล์ต้นฉบับเรียบร้อย: {deleted_count}/{len(files)} ไฟล์")
    
    except Exception as e:
        messagebox.showerror("Cleanup Error", f"เกิดข้อผิดพลาด: {e}")

# GUI Setup
window = tk.Tk()
window.title("🎵 Multi-Platform Audio Transcription (Batch Processing)")
window.geometry("700x900")  # เพิ่มความสูง

# API Key Section
api_frame = tk.Frame(window)
api_frame.pack(pady=5)

# Transcription Method Selection
method_frame = tk.Frame(window)
method_frame.pack(pady=10)
tk.Label(method_frame, text="เลือกวิธีการถอดเสียง:", font=("Arial", 12, "bold")).pack()

transcription_method = tk.StringVar(value="enhanced")

# Radio buttons for transcription methods
methods_info = [
    ("enhanced", "OpenAI Enhanced (แนะนำ) - แม่นยำสูงสุด", "#4CAF50"),
    ("openai", "OpenAI Standard - แม่นยำดี เร็ว", "#2196F3"),
    ("local", f"Whisper Local (ฟรี) - {'พร้อมใช้' if WHISPER_LOCAL_AVAILABLE else 'ต้องติดตั้ง'}", "#FF9800")
]

for value, text, color in methods_info:
    rb = tk.Radiobutton(
        method_frame, 
        text=text, 
        variable=transcription_method, 
        value=value,
        font=("Arial", 10),
        fg=color,
        activeforeground=color
    )
    rb.pack(anchor=tk.W, padx=20)

# URL Input Section
url_frame = tk.Frame(window)
url_frame.pack(pady=5)

tk.Label(url_frame, text="🔗 ใส่ลิงก์ (หลายลิงก์ได้):", font=("Arial", 12, "bold")).pack()
tk.Label(url_frame, text="🔴 YouTube • ⚫ TikTok • 🌐 อื่นๆ | วางทีละบรรทัดหรือคั่นด้วยเว้นวรรค", font=("Arial", 9), fg="gray").pack()

# เปลี่ยนจาก Entry เป็น Text widget เพื่อรองรับหลายบรรทัด
url_entry = tk.Text(window, width=80, height=4, font=("Arial", 10))
url_entry.pack(pady=5)

# เพิ่ม placeholder text
placeholder_text = """วางลิงก์ที่นี่ (รองรับหลายลิงก์):
https://www.youtube.com/watch?v=...
https://vm.tiktok.com/...
https://www.youtube.com/shorts/..."""

url_entry.insert("1.0", placeholder_text)
url_entry.bind('<FocusIn>', lambda e: url_entry.delete("1.0", tk.END) if url_entry.get("1.0", tk.END).strip().startswith("วาง") else None)

download_btn = tk.Button(window, text="🎯 โหลดและถอดเสียงทุกลิงก์", command=on_download_click, bg="#4CAF50", fg="black", font=("Arial", 12, "bold"))
download_btn.pack(pady=10)

status_label = tk.Label(window, text="", font=("Arial", 10))
status_label.pack()

tk.Label(window, text="ผลถอดเสียง (Transcript):", font=("Arial", 12)).pack(pady=5)
transcript_text = ScrolledText(window, width=80, height=10, state=tk.DISABLED)  # เพิ่มความสูง
transcript_text.pack()

tk.Label(window, text="เลือก segment ที่ต้องการตัดเสียง:", font=("Arial", 12)).pack(pady=5)
segments_listbox = tk.Listbox(window, width=80, height=10)  # เพิ่มความสูง
segments_listbox.pack()

cut_btn = tk.Button(window, text="✂️ ตัดและบันทึกเสียง", command=on_cut_click, bg="#2196F3", fg="black", font=("Arial", 12))
cut_btn.pack(pady=10)

# เพิ่มปุ่มสำหรับลบไฟล์ต้นฉบับ
cleanup_btn = tk.Button(window, text="🗑️ ลบไฟล์ต้นฉบับทั้งหมด", command=cleanup_original_files, bg="#FF5722", fg="black", font=("Arial", 10))
cleanup_btn.pack(pady=5)

# Info Labels
info_frame = tk.Frame(window)
info_frame.pack(pady=5)


segments_global = []
all_results_global = []
current_audio_path = ""

window.mainloop()