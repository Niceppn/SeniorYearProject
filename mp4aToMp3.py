import os
import subprocess
from pathlib import Path

def convert_mp4a_to_mp3(source_folder, output_folder):
    """
    แปลงไฟล์ mp4a ทั้งหมดในโฟลเดอร์ source เป็น mp3 และบันทึกในโฟลเดอร์ output
    """
    # ตรวจสอบว่ามี ffmpeg หรือไม่
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ ไม่พบ ffmpeg ในระบบ กรุณาติดตั้ง ffmpeg ก่อน")
        print("สำหรับ macOS: brew install ffmpeg")
        print("สำหรับ Windows: ดาวน์โหลดจาก https://ffmpeg.org/download.html")
        return False
    
    # สร้างโฟลเดอร์ output หากไม่มี
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"✅ สร้างโฟลเดอร์ {output_folder} แล้ว")
    
    # ตรวจสอบโฟลเดอร์ source
    if not os.path.exists(source_folder):
        print(f"❌ ไม่พบโฟลเดอร์ {source_folder}")
        return False
    
    # ค้นหาไฟล์ mp4a ทั้งหมด
    mp4a_files = []
    for file in os.listdir(source_folder):
        if file.lower().endswith('.mp4a') or file.lower().endswith('.m4a'):
            mp4a_files.append(file)
    
    if not mp4a_files:
        print(f"❌ ไม่พบไฟล์ .mp4a หรือ .m4a ในโฟลเดอร์ {source_folder}")
        return False
    
    print(f"🎵 พบไฟล์ที่ต้องแปลง {len(mp4a_files)} ไฟล์:")
    for i, file in enumerate(mp4a_files, 1):
        print(f"  {i}. {file}")
    
    # เริ่มแปลงไฟล์
    success_count = 0
    error_count = 0
    
    for i, file in enumerate(mp4a_files, 1):
        print(f"\n{'='*60}")
        print(f"🔄 กำลังแปลงไฟล์ที่ {i}/{len(mp4a_files)}: {file}")
        print(f"{'='*60}")
        
        # สร้าง path ของไฟล์ต้นฉบับและไฟล์ปลายทาง
        source_path = os.path.join(source_folder, file)
        filename_without_ext = os.path.splitext(file)[0]
        output_path = os.path.join(output_folder, f"{filename_without_ext}.mp3")
        
        try:
            # คำสั่ง ffmpeg สำหรับแปลง mp4a/m4a เป็น mp3
            cmd = [
                'ffmpeg',
                '-i', source_path,           # ไฟล์ input
                '-acodec', 'mp3',           # codec เสียง mp3
                '-ab', '192k',              # bitrate 192 kbps
                '-ar', '44100',             # sample rate 44.1 kHz
                '-y',                       # overwrite output file
                output_path                 # ไฟล์ output
            ]
            
            # รันคำสั่งและซ่อน output ของ ffmpeg
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                # ตรวจสอบขนาดไฟล์
                source_size = os.path.getsize(source_path)
                output_size = os.path.getsize(output_path)
                
                print(f"✅ แปลงสำเร็จ: {file}")
                print(f"   📁 ขนาดต้นฉบับ: {source_size:,} bytes")
                print(f"   📁 ขนาดหลังแปลง: {output_size:,} bytes")
                print(f"   💾 บันทึกที่: {output_path}")
                success_count += 1
            else:
                print(f"❌ แปลงไม่สำเร็จ: {file}")
                print(f"   ข้อผิดพลาด: {result.stderr}")
                error_count += 1
                
        except Exception as e:
            print(f"❌ เกิดข้อผิดพลาดกับไฟล์ {file}: {str(e)}")
            error_count += 1
    
    # สรุปผลการแปลง
    print(f"\n{'='*60}")
    print("🎉 สรุปผลการแปลง")
    print(f"{'='*60}")
    print(f"✅ แปลงสำเร็จ: {success_count} ไฟล์")
    print(f"❌ แปลงไม่สำเร็จ: {error_count} ไฟล์")
    print(f"📂 ไฟล์ที่แปลงแล้วอยู่ที่: {output_folder}")
    print(f"{'='*60}")
    
    return success_count > 0

def main():
    """
    ฟังก์ชันหลักสำหรับแปลงไฟล์ mp4a เป็น mp3
    """
    print("🎵 โปรแกรมแปลง MP4A/M4A เป็น MP3 🎵")
    print("="*50)
    
    # กำหนดโฟลเดอร์ source และ output
    source_folder = "mp4a"
    output_folder = "converted"
    
    print(f"📂 โฟลเดอร์ต้นฉบับ: {source_folder}")
    print(f"📂 โฟลเดอร์ปลายทาง: {output_folder}")
    print("-" * 50)
    
    # เริ่มการแปลง
    success = convert_mp4a_to_mp3(source_folder, output_folder)
    
    if success:
        print("\n🎉 การแปลงเสร็จสิ้น!")
    else:
        print("\n💔 การแปลงไม่สำเร็จ กรุณาตรวจสอบข้อผิดพลาด")

def check_ffmpeg():
    """
    ตรวจสอบว่ามี ffmpeg ในระบบหรือไม่
    """
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        if result.returncode == 0:
            # แยกเอาเฉพาะบรรทัดแรกที่มีข้อมูลเวอร์ชัน
            version_line = result.stdout.split('\n')[0]
            print(f"✅ พบ ffmpeg: {version_line}")
            return True
        else:
            print("❌ ffmpeg ไม่ทำงานปกติ")
            return False
    except FileNotFoundError:
        print("❌ ไม่พบ ffmpeg ในระบบ")
        print("📋 วิธีติดตั้ง:")
        print("   macOS: brew install ffmpeg")
        print("   Windows: ดาวน์โหลดจาก https://ffmpeg.org/download.html")
        print("   Linux: sudo apt install ffmpeg (Ubuntu/Debian)")
        return False

if __name__ == "__main__":
    print("🔧 ตรวจสอบ ffmpeg...")
    if check_ffmpeg():
        print()
        main()
    else:
        print("\n💡 กรุณาติดตั้ง ffmpeg ก่อนใช้งานโปรแกรม")
