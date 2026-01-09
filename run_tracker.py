"""
Aircraft Tracker Runner
Main script to execute YOLOv8 + DeepSORT tracking on videos or webcam
"""
import sys
from pathlib import Path
from yolov8_deepsort import get_tracker


def print_banner():
    """Print application banner"""
    print("\n" + "=" * 70)
    print("           YOLOv8 + DeepSORT Aircraft Tracker")
    print("=" * 70)


def select_video_file():
    """Interactive video file selection"""
    print("\n" + "-" * 70)
    print("VIDEO FILE SELECTION")
    print("-" * 70)
    
    video_path = input("\nEnter video path (or drag and drop): ").strip().strip('"')
    
    # Convert to Path object
    video_path_obj = Path(video_path)
    
    # If relative path, try multiple locations
    if not video_path_obj.is_absolute():
        base_dir = Path(__file__).parent
        
        # Try different possible locations
        possible_paths = [
            base_dir / video_path,
            base_dir / "Sample videos" / video_path,
            Path(video_path)
        ]
        
        for path in possible_paths:
            if path.exists():
                video_path_obj = path
                break
    
    # Check if file exists
    if not video_path_obj.exists():
        print(f"\n❌ Error: Video file not found: {video_path_obj}")
        print("\nPlease check the path and try again.")
        return None
    
    print(f"\n✓ Video found: {video_path_obj}")
    return str(video_path_obj)


def configure_tracker():
    """Configure tracker parameters"""
    print("\n" + "-" * 70)
    print("TRACKER CONFIGURATION")
    print("-" * 70)
    
    # Confidence threshold
    print("\nConfidence Threshold (0.0 - 1.0)")
    print("  - Higher values: Fewer detections, more precision")
    print("  - Lower values: More detections, may include false positives")
    conf_input = input("Enter confidence threshold [default: 0.25]: ").strip()
    
    try:
        conf_threshold = float(conf_input) if conf_input else 0.25
        if not 0.0 <= conf_threshold <= 1.0:
            print("Invalid range. Using default: 0.25")
            conf_threshold = 0.25
    except ValueError:
        print("Invalid input. Using default: 0.25")
        conf_threshold = 0.25
    
    return conf_threshold


def process_video_mode():
    """Video processing mode"""
    print_banner()
    print("\n📹 VIDEO PROCESSING MODE")
    
    # Select video
    video_path = select_video_file()
    if video_path is None:
        return
    
    # Configure tracker
    conf_threshold = configure_tracker()
    
    # Display option
    print("\n" + "-" * 70)
    display_input = input("\nDisplay video during processing? (y/n) [default: y]: ").strip().lower()
    display = display_input != 'n'
    
    # Create tracker
    print("\n" + "-" * 70)
    print("INITIALIZING TRACKER...")
    print("-" * 70)
    
    tracker = get_tracker(conf_threshold=conf_threshold)
    
    # Process video
    print("\n" + "-" * 70)
    print("PROCESSING VIDEO...")
    print("-" * 70)
    
    results = tracker.process_video(video_path, display=display)
    
    if results:
        print("\n" + "=" * 70)
        print("✓ VIDEO PROCESSING COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print(f"Frames Processed: {results['frames_processed']}")
        print(f"Total Aircraft Tracks: {results['total_tracks']}")
        print(f"Output Video: {results['output_path']}")
        print("=" * 70 + "\n")


def process_webcam_mode():
    """Webcam processing mode"""
    print_banner()
    print("\n📷 WEBCAM PROCESSING MODE")
    
    # Camera selection
    print("\n" + "-" * 70)
    print("CAMERA SELECTION")
    print("-" * 70)
    camera_input = input("\nEnter camera ID [default: 0]: ").strip()
    
    try:
        camera_id = int(camera_input) if camera_input else 0
    except ValueError:
        print("Invalid input. Using default camera (0)")
        camera_id = 0
    
    # Configure tracker
    conf_threshold = configure_tracker()
    
    # Create tracker
    print("\n" + "-" * 70)
    print("INITIALIZING TRACKER...")
    print("-" * 70)
    
    tracker = get_tracker(conf_threshold=conf_threshold)
    
    # Process webcam
    print("\n" + "-" * 70)
    print("PROCESSING WEBCAM FEED...")
    print("-" * 70)
    print("Press 'q' in the video window to stop")
    
    results = tracker.process_webcam(camera_id=camera_id)
    
    if results:
        print("\n" + "=" * 70)
        print("✓ WEBCAM PROCESSING COMPLETED")
        print("=" * 70)
        print(f"Frames Processed: {results['frames_processed']}")
        print(f"Total Aircraft Tracks: {results['total_tracks']}")
        print("=" * 70 + "\n")


def batch_process_mode():
    """Batch process multiple videos"""
    print_banner()
    print("\n📂 BATCH PROCESSING MODE")
    
    # Get sample videos directory
    base_dir = Path(__file__).parent
    sample_videos_dir = base_dir / "Sample videos"
    
    if not sample_videos_dir.exists():
        print(f"\n❌ Error: Sample videos directory not found: {sample_videos_dir}")
        return
    
    # Find all video files
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
    video_files = []
    for ext in video_extensions:
        video_files.extend(sample_videos_dir.glob(f"*{ext}"))
    
    if not video_files:
        print(f"\n❌ No video files found in: {sample_videos_dir}")
        return
    
    print(f"\nFound {len(video_files)} video file(s):")
    for i, video in enumerate(video_files, 1):
        print(f"  {i}. {video.name}")
    
    # Configure tracker
    conf_threshold = configure_tracker()
    
    # Display option
    print("\n" + "-" * 70)
    display_input = input("\nDisplay videos during processing? (y/n) [default: n]: ").strip().lower()
    display = display_input == 'y'
    
    # Create tracker
    print("\n" + "-" * 70)
    print("INITIALIZING TRACKER...")
    print("-" * 70)
    
    tracker = get_tracker(conf_threshold=conf_threshold)
    
    # Process each video
    print("\n" + "=" * 70)
    print(f"BATCH PROCESSING {len(video_files)} VIDEO(S)")
    print("=" * 70)
    
    for i, video_path in enumerate(video_files, 1):
        print(f"\n[{i}/{len(video_files)}] Processing: {video_path.name}")
        print("-" * 70)
        
        results = tracker.process_video(str(video_path), display=display)
        
        if results:
            print(f"✓ Completed: {video_path.name}")
        else:
            print(f"❌ Failed: {video_path.name}")
    
    print("\n" + "=" * 70)
    print("✓ BATCH PROCESSING COMPLETED")
    print("=" * 70 + "\n")


def main():
    """Main application entry point"""
    try:
        print_banner()
        
        print("\nSELECT MODE:")
        print("  1. Process single video file")
        print("  2. Process webcam feed")
        print("  3. Batch process videos from 'Sample videos' folder")
        print("  4. Exit")
        print("-" * 70)
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            process_video_mode()
        elif choice == "2":
            process_webcam_mode()
        elif choice == "3":
            batch_process_mode()
        elif choice == "4":
            print("\nExiting... Goodbye!")
            return
        else:
            print("\n❌ Invalid choice. Please select 1, 2, 3, or 4.")
            return
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Processing interrupted by user")
        print("Exiting... Goodbye!\n")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
