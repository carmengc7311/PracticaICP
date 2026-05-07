import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import RegistrationLibrary_Garcia_Carmen as RegistrationLibrary

# 1. Copia aquí tu función generate_registration_animation
#    (la misma que usas en ICP_2D)
def transform_points(points, transformation):
    dim = points.shape[1]
    homogeneous_points = np.hstack([points, np.ones((points.shape[0], 1))])
    transformed_points = homogeneous_points @ transformation.T
    return transformed_points[:, :dim]

def generate_registration_animation(target, source, history):
    fig, ax = plt.subplots(1, 1)
    ax.set_title("ICP Iterations")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    
    margin_factor = 0.5
    
    # Calculate the min and max of source and target points
    x_min = min(np.min(source[:, 0]), np.min(target[:, 0]))
    x_max = max(np.max(source[:, 0]), np.max(target[:, 0]))
    y_min = min(np.min(source[:, 1]), np.min(target[:, 1]))
    y_max = max(np.max(source[:, 1]), np.max(target[:, 1]))
    
    # Add margin to the x and y limits
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    # Apply margin to x and y limits
    ax.set_xlim([x_min - margin_factor * x_range, x_max + margin_factor * x_range])
    ax.set_ylim([y_min - margin_factor * y_range, y_max + margin_factor * y_range])

    # Force equal aspect ratio:
    ax.set_aspect('equal')
    
    # Plot target:
    ax.scatter(target[:, 0], target[:, 1],
               s=4, color="red", label="Target Points")

    # Animated plot source:
    animated_plot, = ax.plot(source[:, 0], source[:, 1],
                             'bo',
                             markersize=2, 
                             label="Transformed Source Points")
    
    # Text box:
    info_box = ax.text(0.05, 0.95, '',
                       transform=ax.transAxes, fontsize=12,
                       verticalalignment='top',
                       bbox=dict(facecolor='white', alpha=0.5))


    def update(i):
        if i != 0:
            metric = history[i-1][0]
            info_box.set_text(f"Iteration: {i}\nRMSE: {metric:.4f}")
    
            transformation = history[i-1][1]  
            transformed_points = transform_points(source, transformation)
            animated_plot.set_data(transformed_points[:, 0], transformed_points[:, 1])
        return animated_plot, info_box
    ani = animation.FuncAnimation(fig, update, frames=len(history)+1,  interval=500)
    plt.legend(loc="lower center")
    
    ani.save("Registration2DAnimation.gif", writer='pillow')
    #ani.save("Registration2DAnimation.mp4")
    
# 2. Define el test mínimo
def main():
    target = np.array([
        [0, 0],
        [1, 0],
        [0, 1]
    ])

    theta = np.pi / 4
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])
    t = np.array([2, 1])

    source = (target @ R.T) + t

    # Ejecutar ICP
    T_est, history = RegistrationLibrary.icp(target, source)

    # Mostrar animación
    generate_registration_animation(target, source, history)

    # Mostrar matrices
    print("Transformación real:")
    print(np.array([
        [np.cos(theta), -np.sin(theta), t[0]],
        [np.sin(theta),  np.cos(theta), t[1]],
        [0, 0, 1]
    ]))

    print("\nTransformación estimada (inversa de T_est):")
    print(np.linalg.inv(T_est))

if __name__ == "__main__":
    main()
