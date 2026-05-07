# -*- coding: utf-8 -*-


# Change GeometryLibrary_Apellido_Nombre by your own library.
import GeometryLibrary_Garcia_Carmen as GeometryLibrary
import sys
import time
import threading
import numpy as np
import open3d as o3d
from PyQt6.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout, 
    QHBoxLayout, QLineEdit, QGroupBox, QSpacerItem, 
    QSizePolicy
)

CLOUD_NAME = "PointCloud"
MESH_NAME = "Mesh"

class GeometriesWindow:
    def __init__(self):
        self.point_cloud = None
        self.update_point_cloud= False
        self.mesh = None
        self.update_mesh = None
        self.main_vis = None
        self.is_done = False
        
    def run(self):
        self.app = o3d.visualization.gui.Application.instance
        self.app.initialize()
        self.main_vis = o3d.visualization.O3DVisualizer("Open3D Viewer",
                                                        1024, 1000)
        self.main_vis.show_skybox(False)
        bg_color = np.array([0.0, 0.0, 0.0, 1], dtype=np.float32)
        self.main_vis.show_ground = True

        self.main_vis.set_background(bg_color, None)
        self.main_vis.set_on_close(self.on_main_window_closing)
        self.app.add_window(self.main_vis)
        threading.Thread(target=self.update_thread, daemon=True).start()
        
        
        
        # Camera configuration:
        camera_position = np.array([100.0, 100.0, 100.0])  # Ajustar según tus necesidades
        target_position = np.array([0,0,0])
        up_position = np.array([0,1,0])
        self.main_vis.setup_camera(1, target_position,camera_position, up_position)
        
        
        self.app.run()

    def update_thread(self):
        while not self.is_done:
            time.sleep(0.1)
            
            def update_cloud():
                    self.main_vis.remove_geometry(CLOUD_NAME)
                    self.main_vis.add_geometry(CLOUD_NAME, self.point_cloud)
                    
            def update_mesh():
                    self.main_vis.remove_geometry(MESH_NAME)
                    self.main_vis.add_geometry(MESH_NAME, self.mesh)

            if self.is_done:  # might have changed while sleeping
                break
            
            if (self.update_point_cloud):
                    self.update_point_cloud = False
                    o3d.visualization.gui.Application.instance.post_to_main_thread(
                        self.main_vis, update_cloud)
            if (self.update_mesh):
                    self.update_mesh = False
                    o3d.visualization.gui.Application.instance.post_to_main_thread(
                        self.main_vis, update_mesh)
                    
    def on_main_window_closing(self):
        self.is_done = True
        return True  # False would cancel the close
    
    def close_window(self):
        self.app.quit()
        
    def _set_point_cloud(self, point_cloud):
        self.point_cloud = point_cloud
        self.update_point_cloud = True

    def _set_mesh(self, mesh):
        self.mesh = mesh
        self.update_mesh = True
        
    def execute_load_point_cloud(self, path):
        try:
            mesh = o3d.io.read_triangle_mesh(path)
    
            if not mesh.has_vertices():
                print(f"Error: No point cloud found in {path}")
                return
    
            pcd = o3d.geometry.PointCloud()
            pcd.points = mesh.vertices
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            self._set_point_cloud(pcd)
            print(f"Loaded point cloud: {path}.")
    
        except Exception as e:
            print(f"Failed to load point cloud: {e}")
    def execute_load_mesh(self, path):
        try:
            mesh = o3d.io.read_triangle_mesh(path)

            if not mesh.has_vertices():
                print(f"Error: No mesh vertices found in {path}")
                return
    
            if not mesh.has_triangles():
                print(f"Warning: Mesh loaded without triangles. Only vertices will be shown.")
            
            self._set_mesh(mesh)
            print(f"Loaded mesh: {path}.")

        except Exception as e:
            print(f"Failed to load mesh: {e}")
            
    def execute_translate_point_cloud(self, translation):
        if (self.point_cloud):
            self._set_point_cloud(
                GeometryLibrary.translate_point_cloud(self.point_cloud, translation)
                )
        else:
            print(f"There is no point cloud to translate.")
            
    def execute_rotate_point_cloud(self, rotation):
        if (self.point_cloud):
            self._set_point_cloud(
                GeometryLibrary.rotate_point_cloud(self.point_cloud, rotation)
                )
        else:
            print(f"There is no point cloud to rotate.")
    def execute_color_point_cloud(self, color):
        if (self.point_cloud):
            self._set_point_cloud(GeometryLibrary.color_point_cloud(
                self.point_cloud, color))
        else:
            print(f"There is no point cloud to color.")
    
    def execute_color_density_point_cloud(self, distance, min_density, max_density):
        if (self.point_cloud):
            self._set_point_cloud(GeometryLibrary.color_point_cloud_with_density(
                self.point_cloud,
                distance,
                min_density, 
                max_density)
                )
        else:
            print(f"There is no point cloud to color.")

    def execute_translate_mesh(self, translation):
        if (self.mesh):
            self._set_mesh(GeometryLibrary.translate_mesh(self.mesh, translation))
        else:
            print(f"There is no mesh to translate.")
            
    def execute_rotate_mesh(self, rotation):
        if (self.mesh):
            self._set_mesh(GeometryLibrary.rotate_mesh(self.mesh, rotation))
        else:
            print(f"There is no mesh to rotate.")
       
    def execute_color_mesh(self, color):
        if (self.mesh):
            self._set_mesh(GeometryLibrary.color_mesh(self.mesh, color))
        else:
            print(f"There is no mesh to color.")
    def execute_triangle_normals_mesh(self):
        self._set_mesh(GeometryLibrary.compute_mesh_vertex_normals(self.mesh))

class InputWidgetXYZ(QWidget):
    def __init__(self,
                 button_text,
                 placeHolder0 = "X",
                 placeHolder1 = "Y",
                 placeHolder2 = "Z"):
        super().__init__()

        # Create main vertical layout
        main_layout = QVBoxLayout()

        # Horizontal layout for the text fields (Middle)
        text_layout = QHBoxLayout()
        self.x_input = QLineEdit()
        self.y_input = QLineEdit()
        self.z_input = QLineEdit()

        self.x_input.setPlaceholderText(placeHolder0)
        self.y_input.setPlaceholderText(placeHolder1)
        self.z_input.setPlaceholderText(placeHolder2)

        text_layout.addWidget(self.x_input)
        text_layout.addWidget(self.y_input)
        text_layout.addWidget(self.z_input)

        main_layout.addLayout(text_layout)

        # Bottom button
        self.button = QPushButton(button_text)
        main_layout.addWidget(self.button)

        # Set main layout
        self.setLayout(main_layout)
        
    def getInput(self):
        try:
            # Try to get and convert each input to a float
            x = float(self.x_input.text()) if self.x_input.text() else 0.0
        except ValueError:
            print("Invalid value for X, setting to 0.")
            x = 0.0
            
        try:
            y = float(self.y_input.text()) if self.y_input.text() else 0.0
        except ValueError:
            print("Invalid value for Y, setting to 0.")
            y = 0.0
            
        try:
            z = float(self.z_input.text()) if self.z_input.text() else 0.0
        except ValueError:
            print("Invalid value for Z, setting to 0.")
            z = 0.0
        
        return np.array([x, y, z])

class Widgets(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Main layout (splitting the UI into two parts)
        main_layout = QHBoxLayout()

        # Create group boxes with titles
        self.group_point_cloud = QGroupBox("Point Cloud")
        self.group_mesh = QGroupBox("Mesh")
        
        # Layouts inside group boxes
        self.layout_point_cloud = QVBoxLayout()
        self.layout_mesh = QVBoxLayout()
        
        # Point Cloud Widgets:
        self.btn_load_point_cloud = QPushButton("Load From File")
        self.edit_load_point_cloud = QLineEdit("rabbit.ply")
        self.translate_point_cloud = InputWidgetXYZ("Translate")
        self.rotate_point_cloud = InputWidgetXYZ("Rotate")
        self.color_point_cloud = InputWidgetXYZ("Color", "R", "G", "B")
        self.color_density_point_cloud = InputWidgetXYZ("Color Density", "Distance", "Min Density", "Max Density")

        # Mesh Widgets:
        self.btn_load_mesh = QPushButton("Load From File")
        self.edit_load_mesh = QLineEdit("dragon.ply")
        self.translate_mesh = InputWidgetXYZ("Translate")
        self.rotate_mesh = InputWidgetXYZ("Rotate")
        self.color_mesh = InputWidgetXYZ("Color", "R", "G", "B")
        self.btn_triangle_normals_mesh = QPushButton("Compute Triangle Normals")
        
        # Add widgets to point cloud layout:
        self.layout_point_cloud.addItem(QSpacerItem(20, 40,
                                                    QSizePolicy.Policy.Expanding,
                                                    QSizePolicy.Policy.Expanding))
        self.layout_point_cloud.addWidget(self.edit_load_point_cloud)
        self.layout_point_cloud.addWidget(self.btn_load_point_cloud)
        self.layout_point_cloud.addWidget(self.translate_point_cloud)
        self.layout_point_cloud.addWidget(self.rotate_point_cloud)
        self.layout_point_cloud.addWidget(self.color_point_cloud)
        self.layout_point_cloud.addWidget(self.color_density_point_cloud)
        self.layout_point_cloud.addItem(QSpacerItem(20, 40,
                                                    QSizePolicy.Policy.Expanding,
                                                    QSizePolicy.Policy.Expanding))

        # Add widgets to mesh layout:
        self.layout_mesh.addItem(QSpacerItem(20, 40,
                                             QSizePolicy.Policy.Expanding,
                                             QSizePolicy.Policy.Expanding))
        self.layout_mesh.addWidget(self.edit_load_mesh)
        self.layout_mesh.addWidget(self.btn_load_mesh)
        self.layout_mesh.addWidget(self.translate_mesh)
        self.layout_mesh.addWidget(self.rotate_mesh)
        self.layout_mesh.addWidget(self.color_mesh)
        self.layout_mesh.addWidget(self.btn_triangle_normals_mesh)
        self.layout_mesh.addItem(QSpacerItem(20, 40,
                                             QSizePolicy.Policy.Expanding,
                                             QSizePolicy.Policy.Expanding))

        # Add layouts to groups:
        self.group_point_cloud.setLayout(self.layout_point_cloud)
        self.group_mesh.setLayout(self.layout_mesh)

        # Add groups to main_layout:
        main_layout.addWidget(self.group_point_cloud, 1)
        main_layout.addWidget(self.group_mesh, 1)

        self.setLayout(main_layout)
        
        self.make_widget_connections()
        
    def make_widget_connections(self):
        self.btn_load_point_cloud.clicked.connect(self.call_load_point_cloud)
        self.translate_point_cloud.button.clicked.connect(self.call_translate_point_cloud)
        self.rotate_point_cloud.button.clicked.connect(self.call_rotate_point_cloud)
        self.color_point_cloud.button.clicked.connect(self.call_color_point_cloud)
        self.color_density_point_cloud.button.clicked.connect(self.call_color_density_point_cloud)
        
        self.btn_load_mesh.clicked.connect(self.call_load_mesh)
        self.translate_mesh.button.clicked.connect(self.call_translate_mesh)
        self.rotate_mesh.button.clicked.connect(self.call_rotate_mesh)
        self.color_mesh.button.clicked.connect(self.call_color_mesh)
        self.btn_triangle_normals_mesh.clicked.connect(self.call_triangle_normals_mesh)
        
    def call_load_point_cloud(self):
        path = self.edit_load_point_cloud.text()
        self.parent().geometriesWindow.execute_load_point_cloud(path)
        
    def call_load_mesh(self):
        path = self.edit_load_mesh.text()
        self.parent().geometriesWindow.execute_load_mesh(path)
        
    def call_translate_point_cloud(self):
        translation = self.translate_point_cloud.getInput()
        if (translation is not None):
            self.parent().geometriesWindow.execute_translate_point_cloud(translation)
            
    def call_rotate_point_cloud(self):
        rotation = self.rotate_point_cloud.getInput()
        if (rotation is not None):
            self.parent().geometriesWindow.execute_rotate_point_cloud(rotation)
    def call_color_point_cloud(self):
        color = self.color_point_cloud.getInput()
        if (color is not None):
            self.parent().geometriesWindow.execute_color_point_cloud(color)
    
    def call_color_density_point_cloud(self):
        color_density_input = self.color_density_point_cloud.getInput()
        if (color_density_input is not None):
            self.parent().geometriesWindow.execute_color_density_point_cloud(
                color_density_input[0], color_density_input[1], color_density_input[2] )
            
    def call_translate_mesh(self):
        translation = self.translate_mesh.getInput()
        if (translation is not None):
            self.parent().geometriesWindow.execute_translate_mesh(translation)
            
    def call_rotate_mesh(self):
        rotation = self.rotate_mesh.getInput()
        if (rotation is not None):
            self.parent().geometriesWindow.execute_rotate_mesh(rotation)
            
    def call_color_mesh(self):
        color = self.color_mesh.getInput()
        if (color is not None):
            self.parent().geometriesWindow.execute_color_mesh(color)
            
    def call_triangle_normals_mesh(self):
        self.parent().geometriesWindow.execute_triangle_normals_mesh()
            
class QtWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("UI Controller")
        self.setGeometry(0, 0, 900, 1000)
        layout = QVBoxLayout()
        self.widgets = Widgets(self)
        layout.addWidget(self.widgets)
        self.setLayout(layout)
        
        self.geometriesWindow = GeometriesWindow()
        
    def launch_open3d(self):
        self.geometriesWindow.run()

        
    def closeEvent(self, event):
        self.geometriesWindow.close_window()
        event.accept()  # Accept the event to close the window

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = QtWindow()
    window.show()
    window.launch_open3d()
    app.closeAllWindows()    
