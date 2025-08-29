#!/usr/bin/env python3
"""
MLX Knife Demo Recorder
Automatisiert Screen Recording mit ffmpeg für MLX Knife Web UI Demos

Requirements:
- ffmpeg installiert (brew install ffmpeg)
- MLX Knife server läuft auf localhost:8000
- Optional: playwright für Browser-Automation (pip install playwright)
"""

import subprocess
import time
import sys
from pathlib import Path

class MLXDemoRecorder:
    def __init__(self, output_name="mlx_demo", duration=120):
        self.output_name = output_name
        self.duration = duration
        self.ffmpeg_process = None
        
    def start_recording(self, include_audio=True):
        """Start ffmpeg screen recording"""
        print("🎬 MLX Knife Demo Recorder")
        print(f"📹 Recording: {self.duration} seconds")
        print(f"📁 Output: {self.output_name}.mp4")
        
        # ffmpeg command for macOS screen recording
        cmd = [
            'ffmpeg',
            '-f', 'avfoundation',
            '-i', '1:0' if include_audio else '1',  # Screen + optional audio
            '-r', '30',                              # 30 FPS
            '-t', str(self.duration),                # Duration
            '-vf', 'scale=1280:720',                 # Scale for web
            '-y',                                    # Overwrite existing
            f'{self.output_name}.mp4'
        ]
        
        print("🔴 Starting recording in 3 seconds...")
        print("   Make sure MLX Knife server is running!")
        time.sleep(3)
        
        self.ffmpeg_process = subprocess.Popen(
            cmd, 
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.PIPE
        )
        
        return self.ffmpeg_process
    
    def show_countdown(self):
        """Show live countdown during recording"""
        try:
            for remaining in range(self.duration, 0, -1):
                print(f"\r⏱️  Recording: {remaining:3d}s remaining", end="", flush=True)
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n⏹️  Recording stopped by user")
            self.stop_recording()
            return False
        
        print(f"\n✅ Recording finished: {self.output_name}.mp4")
        return True
    
    def stop_recording(self):
        """Stop recording gracefully"""
        if self.ffmpeg_process:
            self.ffmpeg_process.terminate()
            self.ffmpeg_process.wait()
    
    def convert_to_gif(self, max_size=800):
        """Convert MP4 to Discord-friendly GIF"""
        if not Path(f"{self.output_name}.mp4").exists():
            print("❌ No MP4 file found to convert")
            return
            
        print("🔄 Converting to GIF for Discord...")
        
        gif_cmd = [
            'ffmpeg',
            '-i', f'{self.output_name}.mp4',
            '-vf', f'fps=10,scale={max_size}:-1:flags=lanczos',
            '-y',
            f'{self.output_name}.gif'
        ]
        
        try:
            subprocess.run(gif_cmd, check=True, capture_output=True)
            print(f"🎯 GIF ready: {self.output_name}.gif")
        except subprocess.CalledProcessError as e:
            print(f"❌ GIF conversion failed: {e}")
    
    def record_demo(self, convert_gif=True):
        """Complete recording workflow"""
        print("=" * 50)
        print("🦫 MLX Knife Demo Recording Session")
        print("=" * 50)
        
        # Pre-flight checks
        if not self._check_ffmpeg():
            return False
        
        if not self._check_server():
            return False
        
        # Start recording
        self.start_recording()
        
        # Show demo instructions
        self._show_demo_instructions()
        
        # Live countdown
        success = self.show_countdown()
        
        if success:
            # Wait for ffmpeg to finish
            self.ffmpeg_process.wait()
            
            # Convert to GIF if requested
            if convert_gif:
                self.convert_to_gif()
            
            self._show_results()
        
        return success
    
    def _check_ffmpeg(self):
        """Check if ffmpeg is installed"""
        try:
            subprocess.run(['ffmpeg', '-version'], 
                         capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ ffmpeg not found! Install with: brew install ffmpeg")
            return False
    
    def _check_server(self):
        """Check if MLX Knife server is running"""
        try:
            import urllib.request
            urllib.request.urlopen("http://localhost:8000", timeout=2)
            print("✅ MLX Knife server is running")
            return True
        except:
            print("❌ MLX Knife server not found!")
            print("   Start with: mlxk server")
            return False
    
    def _show_demo_instructions(self):
        """Show what to do during recording"""
        print("\n🎯 DEMO SCRIPT - Follow these steps:")
        print("   1. Open browser to http://localhost:8000")
        print("   2. Type: 'Tell a story in 100 words'")
        print("   3. Wait for Phi-3 response (~8s)")
        print("   4. Switch model to Mistral-7B")
        print("   5. Type: 'tell again'") 
        print("   6. Wait for different story (~8s)")
        print("   7. Switch to Qwen3, type: 'translate to Thai'")
        print("   8. Switch to Llama-3.3-70B, type: 'translate Thai story to English'")
        print("   9. Enjoy the meta-comment! 🤖")
        print("\n🚀 GO GO GO!")
        print()
    
    def _show_results(self):
        """Show final results"""
        print("\n" + "=" * 50)
        print("🎉 Recording Session Complete!")
        print("=" * 50)
        
        mp4_path = Path(f"{self.output_name}.mp4")
        gif_path = Path(f"{self.output_name}.gif")
        
        if mp4_path.exists():
            size_mb = mp4_path.stat().st_size / (1024*1024)
            print(f"📹 MP4: {mp4_path} ({size_mb:.1f} MB)")
        
        if gif_path.exists():
            size_mb = gif_path.stat().st_size / (1024*1024) 
            print(f"🎯 GIF: {gif_path} ({size_mb:.1f} MB)")
            print("   → Perfect for Discord/Twitter!")
        
        print("\n🦫 Ready for LocalLLM showcase!")


def main():
    """Command line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="MLX Knife Demo Recorder")
    parser.add_argument("--name", default="mlx_thai_demo", 
                       help="Output filename (without extension)")
    parser.add_argument("--duration", type=int, default=120,
                       help="Recording duration in seconds")
    parser.add_argument("--no-gif", action="store_true",
                       help="Skip GIF conversion")
    
    args = parser.parse_args()
    
    recorder = MLXDemoRecorder(args.name, args.duration)
    success = recorder.record_demo(convert_gif=not args.no_gif)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()