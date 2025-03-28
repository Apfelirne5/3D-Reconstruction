import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

class GaussianSplattingReconstructor:
    def __init__(self, ply_path):
        """
        Initialize the reconstructor with the trained PLY file
        
        Args:
            ply_path (str): Path to the trained Gaussian Splatting PLY file
        """
        self.point_cloud = o3d.io.read_point_cloud(ply_path)
        
    def analyze_point_cloud(self):
        """
        Perform comprehensive analysis of the reconstructed point cloud
        """
        points_array = np.asarray(self.point_cloud.points)
        
        print("Point Cloud Analysis:")
        print(f"Total points: {len(self.point_cloud.points)}")
        
        print("\nGeometric Properties:")
        print(f"Bounding Box Min: {points_array.min(axis=0)}")
        print(f"Bounding Box Max: {points_array.max(axis=0)}")
        
        # Calculate point cloud centroid
        centroid = points_array.mean(axis=0)
        print(f"Point Cloud Centroid: {centroid}")
        
        # Point distribution across axes
        plt.figure(figsize=(15, 5))
        
        plt.subplot(131)
        plt.hist(points_array[:, 0], bins=50, edgecolor='black')
        plt.title('X-axis Distribution')
        plt.xlabel('X')
        plt.ylabel('Point Count')
        
        plt.subplot(132)
        plt.hist(points_array[:, 1], bins=50, edgecolor='black')
        plt.title('Y-axis Distribution')
        plt.xlabel('Y')
        
        plt.subplot(133)
        plt.hist(points_array[:, 2], bins=50, edgecolor='black')
        plt.title('Z-axis Distribution')
        plt.xlabel('Z')
        
        plt.tight_layout()
        plt.savefig('point_cloud_axis_distribution.png')
        plt.close()
        
    def advanced_visualization(self, output_path='reconstructed_scene.png'):
        """
        Advanced point cloud visualization with color mapping
        
        Args:
            output_path (str): Path to save the visualization
        """
        # Create visualization
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=1600, height=900)
        
        # Add point cloud
        vis.add_geometry(self.point_cloud)
        
        # Customize rendering
        render_option = vis.get_render_option()
        render_option.background_color = np.asarray([1, 1, 1])  # White background
        render_option.point_size = 1.0  # Adjust point size
        
        # Color point cloud based on depth
        points = np.asarray(self.point_cloud.points)
        depth_colors = plt.cm.viridis((points[:, 2] - points[:, 2].min()) / 
                                       (points[:, 2].max() - points[:, 2].min()))
        self.point_cloud.colors = o3d.utility.Vector3dVector(depth_colors[:, :3])
        vis.update_geometry(self.point_cloud)
        
        # Set camera view
        ctr = vis.get_view_control()
        ctr.set_front([0, -1, -0.5])
        ctr.set_lookat([0, 0, 0])
        ctr.set_up([0, 0, 1])
        
        # Update and capture
        vis.update_renderer()
        vis.poll_events()
        vis.capture_screen_image(output_path)
        vis.destroy_window()
        
        print(f"Visualization saved to {output_path}")

def main():
    # Replace with your actual PLY file path
    ply_path = '/Users/thanushnavaratnam/Desktop/Code/Gaussian_Splatting/bicycle_4_Kopie_5/export.ply'
    
    # Initialize reconstructor
    reconstructor = GaussianSplattingReconstructor(ply_path)
    
    # Analyze point cloud
    reconstructor.analyze_point_cloud()
    
    # Advanced visualization
    reconstructor.advanced_visualization()

if __name__ == '__main__':
    main()
