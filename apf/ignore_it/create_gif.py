import imageio
import os
import glob

def frames_to_gif(frame_dir='simulation_frames', output_filename='simulation.gif', fps=10):
    """
    Convert PNG frames to an animated GIF
    
    Args:
        frame_dir (str): Directory containing the frames
        output_filename (str): Path to save the output GIF
        fps (int): Frames per second for the animation
    """
    # Get all frame files in order
    files = sorted(glob.glob(os.path.join(frame_dir, 'frame_*.png')))
    
    if not files:
        print(f"No frames found in directory: {frame_dir}")
        return
    
    print(f"Found {len(files)} frames. Creating GIF...")
    
    # Read all frames and create GIF
    images = []
    for file in files:
        images.append(imageio.imread(file))
    
    # Save as GIF
    imageio.mimsave(output_filename, images, duration=1/fps)
    print(f"GIF saved as: {output_filename}")

# Example usage
if __name__ == "__main__":
    frames_to_gif(fps=15)  # Adjust fps as needed for smoother/faster animation