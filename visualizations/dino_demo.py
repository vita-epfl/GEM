import os
import pickle  # For saving and loading data
import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import RectangleSelector, Slider


class SelectionTool:
    def __init__(
        self,
        source_frame_paths,
        target_frame_paths,
        resized_height=320,
        resized_width=576,
        grid_height=40,
        grid_width=72,
        n_target_frames=25,
        n_conditions=5,
        max_num_identities=5,
        loaded_data=None,
    ):
        # Assign parameters to instance variables
        self.resized_height = resized_height  # Height to resize images to
        self.resized_width = resized_width  # Width to resize images to
        self.grid_height = grid_height  # Number of tiles along height
        self.grid_width = grid_width  # Number of tiles along width

        self.tile_height = self.resized_height // self.grid_height
        self.tile_width = self.resized_width // self.grid_width

        self.image_width = self.resized_width
        self.image_height = self.resized_height

        self.n_target_frames = n_target_frames  # Number of target frames
        self.n_conditions = n_conditions  # Number of conditions
        self.max_num_identities = max_num_identities

        self.source_frame_paths = source_frame_paths
        self.n_source_frames = len(source_frame_paths)

        self.target_frame_paths = target_frame_paths
        # self.n_target_frames = len(target_frame_paths)  # You could alternatively set it this way if you prefer

        # Load and resize the source frames
        self.source_frames = [
            cv2.resize(
                cv2.cvtColor(cv2.imread(frame_path), cv2.COLOR_BGR2RGB),
                (self.resized_width, self.resized_height),
            )
            for frame_path in self.source_frame_paths
        ]

        # Load and resize the target frames
        self.target_frames = [
            cv2.resize(
                cv2.cvtColor(cv2.imread(frame_path), cv2.COLOR_BGR2RGB),
                (self.resized_width, self.resized_height),
            )
            for frame_path in self.target_frame_paths
        ]

        # Initialize data structures
        self.current_condition = 0
        self.source_frame_indices = [0] * self.n_conditions
        self.target_frame_indices = [0] * self.n_conditions

        # Changed to be per source frame
        self.source_masks = [
            np.zeros((self.grid_height, self.grid_width), dtype=int)
            for _ in range(self.n_source_frames)
        ]
        self.target_masks = [
            np.zeros((self.grid_height, self.grid_width), dtype=int)
            for _ in range(self.n_conditions)
        ]
        # Changed to be per source frame
        self.bounding_boxes = [None] * self.n_source_frames
        self.identities = [None] * self.n_conditions
        self.trajectories = [None] * self.n_conditions  # For storing trajectories

        self.modified_conditions = set()
        self.modified_source_frames = set()

        self.current_mode = "tile"
        self.exit_flag = False

        # Initialize figure and axes
        self.fig = plt.figure(figsize=(18, 7))
        gs = plt.GridSpec(1, 3, width_ratios=[1, 1, 0.1], wspace=0.2)
        self.ax_source = self.fig.add_subplot(gs[0])
        self.ax_target = self.fig.add_subplot(gs[1])
        self.ax_trajectory = self.fig.add_subplot(gs[2])

        self.ax_source.set_title("Source Locations")
        self.ax_target.set_title("Target Locations")
        self.ax_trajectory.set_title("Trajectory")

        # Add instruction text
        instructions = (
            "Instructions:\n"
            "- Press 't' to switch to Tile Selection Mode\n"
            "- Press 'b' to switch to Bounding Box Mode\n"
            "- Press 'r' to switch to Trajectory Mode\n"
            "- Press 'Enter' to finish selection\n"
            "- Press number keys '1' to '5' to assign an Identity\n"
            "- Use sliders to change Condition, Source Frame, and Target Frame\n"
            "- In Tile Mode, click on tiles to select/deselect\n"
            "- In BBox Mode, click and drag to draw a bounding box\n"
            "- In Trajectory Mode, click on the Source Image to select center,\n"
            "  then draw in the Trajectory window\n"
            "- Note: The displayed target image is always the first frame,\n"
            "  but the selected target frame index is still recorded."
        )
        self.fig.text(0.01, 0.95, instructions, fontsize=10, va="top")

        # Add text to display current identity
        self.identity_text = self.fig.text(
            0.5, 0.9, "", fontsize=12, va="center", ha="center", color="blue"
        )

        # Variables for trajectory drawing
        self.drawing = False
        (self.trajectory_line,) = self.ax_trajectory.plot([], [], "b-")  # Initialize an empty line
        self.trajectory_x = []
        self.trajectory_y = []

        self.x_min = 0
        self.y_min = 0
        self.x_max = self.image_width
        self.y_max = self.image_height

        # If loading previously saved data
        if loaded_data is not None:
            condition_numbers = loaded_data["condition_numbers"]
            for idx, cond_idx in enumerate(condition_numbers):
                self.source_frame_indices[cond_idx] = loaded_data["source_indices"][idx]
                self.target_frame_indices[cond_idx] = loaded_data["target_indices"][idx]
                source_idx = self.source_frame_indices[cond_idx]
                self.source_masks[source_idx] = loaded_data["source_masks"][idx]
                self.target_masks[cond_idx] = loaded_data["target_masks"][idx]
                bbox = loaded_data["bounding_boxes"][idx]

                # Convert normalized bounding box back to pixel coordinates
                if not np.array_equal(bbox, [0, 0, 0, 0]):
                    w_norm, h_norm, x_norm, y_norm = bbox
                    x_center = (x_norm + 1) / 2 * self.image_width
                    y_center = (y_norm + 1) / 2 * self.image_height
                    w_pixels = w_norm * self.image_width
                    h_pixels = h_norm * self.image_height
                    x1 = x_center - w_pixels / 2
                    x2 = x_center + w_pixels / 2
                    y1 = y_center - h_pixels / 2
                    y2 = y_center + h_pixels / 2
                    self.bounding_boxes[source_idx] = (x1, y1, x2, y2)
                else:
                    self.bounding_boxes[source_idx] = None

                identity = loaded_data["identities"][idx]
                if identity != -1:
                    self.identities[cond_idx] = identity
                else:
                    self.identities[cond_idx] = None

                trajectory = loaded_data.get("trajectories", [None] * self.n_conditions)[idx]
                self.trajectories[cond_idx] = trajectory

                self.modified_conditions.add(cond_idx)
                self.modified_source_frames.add(source_idx)
            print(f"Loaded data for conditions: {condition_numbers}")

    def run(self):
        """
        Launch the interactive selection tool. Returns the selection data once you press 'Enter'.
        """
        # Create sliders for condition, source frame, and target frame
        ax_slider_condition = plt.axes([0.25, 0.25, 0.5, 0.03], facecolor="lightgrey")
        self.slider_condition = Slider(
            ax_slider_condition,
            "Condition Number",
            0,
            self.n_conditions - 1,
            valinit=0,
            valstep=1,
        )

        ax_slider_source = plt.axes([0.25, 0.15, 0.5, 0.03], facecolor="lightgoldenrodyellow")
        self.slider_source = Slider(
            ax_slider_source,
            "Source Frame",
            0,
            self.n_source_frames - 1,
            valinit=0,
            valstep=1,
        )

        ax_slider_target = plt.axes([0.25, 0.05, 0.5, 0.03], facecolor="lightblue")
        self.slider_target = Slider(
            ax_slider_target,
            "Target Frame",
            0,
            self.n_target_frames - 1,
            valinit=0,
            valstep=1,
        )

        # Draw the initial grid for the first source and first target frames
        source_frame_index = self.source_frame_indices[self.current_condition]
        self.redraw_image(
            self.ax_source,
            self.source_masks[source_frame_index],
            "red",
            self.source_frames[source_frame_index],
        )
        self.redraw_image(
            self.ax_target,
            self.target_masks[self.current_condition],
            "green",
            self.target_frames[0],
        )  # Always display the first target frame

        # Initialize the trajectory plot
        self.load_trajectory()

        # Update the identity display
        self.update_identity_display()

        # Create the RectangleSelector for bounding box mode
        self.rect_selector = RectangleSelector(
            self.ax_source,
            self.onselect,
            useblit=False,
            button=[1],  # Left mouse button
            minspanx=5,
            minspany=5,
            spancoords="pixels",
            interactive=True,
        )
        self.rect_selector.set_active(False)  # Start with selector deactivated

        # Connect events
        self.fig.canvas.mpl_connect("button_press_event", self.on_button_press)
        self.fig.canvas.mpl_connect("button_release_event", self.on_button_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)

        self.slider_condition.on_changed(self.update_condition)
        self.slider_source.on_changed(self.update_source_frame)
        self.slider_target.on_changed(self.update_target_frame)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)

        # Start the event loop
        while not self.exit_flag:
            plt.pause(0.1)

        # Close the plot window
        plt.close(self.fig)

        # After GUI is closed, assemble data and return
        return self.get_data()

    def on_click(self, event):
        """
        Handle click events in 'tile' and 'trajectory' modes.
        """
        if self.current_mode == "tile":
            if event.inaxes == self.ax_source:
                self.on_click_source(event)
            elif event.inaxes == self.ax_target:
                self.on_click_target(event)
        elif self.current_mode == "trajectory":
            if event.inaxes == self.ax_source:
                self.on_click_source(event)

    def on_click_source(self, event):
        """
        Handle clicks on the source image.
        """
        if event.xdata is None or event.ydata is None:
            return  # Ignore clicks outside the axes

        if self.current_mode == "tile":
            # Toggle tile
            x = int(event.xdata // self.tile_width)
            y = int(event.ydata // self.tile_height)

            source_frame_index = self.source_frame_indices[self.current_condition]

            # Toggle tile in the source mask
            self.source_masks[source_frame_index][y, x] = (
                1 - self.source_masks[source_frame_index][y, x]
            )

            # Mark as modified
            self.modified_source_frames.add(source_frame_index)
            self.modified_conditions.add(self.current_condition)

            # Redraw
            self.redraw_image(
                self.ax_source,
                self.source_masks[source_frame_index],
                "red",
                self.source_frames[source_frame_index],
            )
        elif self.current_mode == "trajectory":
            # Update trajectory window based on click
            x = event.xdata
            y = event.ydata
            self.update_trajectory_window(x, y)

    def on_click_target(self, event):
        """
        Handle clicks on the target image in tile mode.
        """
        if event.xdata is None or event.ydata is None:
            return  # Ignore clicks outside the axes

        if self.current_mode == "tile":
            x = int(event.xdata // self.tile_width)
            y = int(event.ydata // self.tile_height)

            # Toggle tile in the target mask
            self.target_masks[self.current_condition][y, x] = (
                1 - self.target_masks[self.current_condition][y, x]
            )

            # Mark as modified
            self.modified_conditions.add(self.current_condition)

            # Redraw
            self.redraw_image(
                self.ax_target,
                self.target_masks[self.current_condition],
                "green",
                self.target_frames[0],  # Always display first target frame
            )

    def onselect(self, eclick, erelease):
        """
        Handle bounding box selection in bbox mode.
        """
        if self.current_mode != "bbox":
            return

        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        if None in (x1, y1, x2, y2):
            return

        # Clamp coordinates to image boundaries
        x1 = max(0, min(x1, self.resized_width))
        x2 = max(0, min(x2, self.resized_width))
        y1 = max(0, min(y1, self.resized_height))
        y2 = max(0, min(y2, self.resized_height))

        source_frame_index = self.source_frame_indices[self.current_condition]
        self.bounding_boxes[source_frame_index] = (x1, y1, x2, y2)

        # Mark as modified
        self.modified_source_frames.add(source_frame_index)
        self.modified_conditions.add(self.current_condition)

        # Redraw
        self.redraw_image(
            self.ax_source,
            self.source_masks[source_frame_index],
            "red",
            self.source_frames[source_frame_index],
        )

    def redraw_image(self, ax, mask, color, frame):
        """
        Redraw the given axes with the provided frame and mask.
        """
        ax.clear()
        ax.imshow(frame)

        # Draw grid
        for i in range(0, self.resized_width, self.tile_width):
            ax.axvline(i, color="white", linewidth=0.5)
        for i in range(0, self.resized_height, self.tile_height):
            ax.axhline(i, color="white", linewidth=0.5)

        # Highlight selected tiles
        for y in range(mask.shape[0]):
            for x in range(mask.shape[1]):
                if mask[y, x] == 1:
                    rect = plt.Rectangle(
                        (x * self.tile_width, y * self.tile_height),
                        self.tile_width,
                        self.tile_height,
                        linewidth=1,
                        edgecolor="none",
                        facecolor=color,
                        alpha=0.5,
                    )
                    ax.add_patch(rect)

        # Draw bounding box if any (on the source only)
        if ax == self.ax_source:
            source_frame_index = self.source_frame_indices[self.current_condition]
            if self.bounding_boxes[source_frame_index] is not None:
                x1, y1, x2, y2 = self.bounding_boxes[source_frame_index]
                rect = plt.Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    linewidth=2,
                    edgecolor="yellow",
                    facecolor="none",
                )
                ax.add_patch(rect)

        self.fig.canvas.draw_idle()

    def update_condition(self, val):
        """
        When the condition slider changes, update the displayed frames/masks.
        """
        self.current_condition = int(val)

        # Sync the source/target sliders without triggering them
        self.slider_source.eventson = False
        self.slider_source.set_val(self.source_frame_indices[self.current_condition])
        self.slider_source.eventson = True

        self.slider_target.eventson = False
        self.slider_target.set_val(self.target_frame_indices[self.current_condition])
        self.slider_target.eventson = True

        source_frame_index = self.source_frame_indices[self.current_condition]

        # Redraw
        self.redraw_image(
            self.ax_source,
            self.source_masks[source_frame_index],
            "red",
            self.source_frames[source_frame_index],
        )
        self.redraw_image(
            self.ax_target,
            self.target_masks[self.current_condition],
            "green",
            self.target_frames[0],
        )
        self.load_trajectory()
        self.update_identity_display()

    def update_source_frame(self, val):
        """
        When the source frame slider changes, update the displayed source image.
        """
        self.source_frame_indices[self.current_condition] = int(val)
        self.modified_conditions.add(self.current_condition)

        source_frame_index = self.source_frame_indices[self.current_condition]
        self.redraw_image(
            self.ax_source,
            self.source_masks[source_frame_index],
            "red",
            self.source_frames[source_frame_index],
        )

    def update_target_frame(self, val):
        """
        When the target frame slider changes, record the chosen target but keep displaying the first target image.
        """
        self.target_frame_indices[self.current_condition] = int(val)
        self.modified_conditions.add(self.current_condition)

        self.redraw_image(
            self.ax_target,
            self.target_masks[self.current_condition],
            "green",
            self.target_frames[0],  # Always display the first target frame
        )

    def on_key_press(self, event):
        """
        Handle key presses for mode switching, identity assignment, and exiting.
        """
        if event.key == "t":
            self.current_mode = "tile"
            self.rect_selector.set_active(False)
            print("Switched to Tile Selection Mode")
        elif event.key == "b":
            self.current_mode = "bbox"
            self.rect_selector.set_active(True)
            print("Switched to Bounding Box Mode")
        elif event.key == "r":
            self.current_mode = "trajectory"
            self.rect_selector.set_active(False)
            print("Switched to Trajectory Mode")
        elif event.key in ["enter", "return"]:
            print("Exiting selection mode.")
            self.exit_flag = True
        elif event.key in [str(i) for i in range(0, self.max_num_identities)]:
            self.assign_identity(int(event.key))
        else:
            pass

    def assign_identity(self, identity_number):
        """
        Assign an identity to the current condition.
        """
        self.identities[self.current_condition] = identity_number
        self.modified_conditions.add(self.current_condition)
        print(f"Assigned Identity {identity_number} to Condition {self.current_condition}")
        self.update_identity_display()

    def update_identity_display(self):
        """
        Update the on-figure text showing the current identity.
        """
        identity = self.identities[self.current_condition]
        if identity is not None:
            identity_text = f"Current Identity: {identity}"
        else:
            identity_text = "No Identity Assigned"
        self.identity_text.set_text(identity_text)
        self.fig.canvas.draw_idle()

    def get_data(self):
        """
        Assemble and return the data for all modified conditions, plus images.
        """
        # Which conditions have been modified?
        modified_indices = sorted(self.modified_conditions)
        num_modified = len(modified_indices)

        # Assemble bounding boxes (normalized)
        bounding_boxes = np.zeros((num_modified, 4), dtype=float)
        for idx, cond_idx in enumerate(modified_indices):
            source_idx = self.source_frame_indices[cond_idx]
            if self.bounding_boxes[source_idx] is not None:
                x1, y1, x2, y2 = self.bounding_boxes[source_idx]
                h = abs(y2 - y1) / self.image_height
                w = abs(x2 - x1) / self.image_width
                x_center = (x1 + x2) / 2
                y_center = (y1 + y2) / 2
                x_norm = 2 * (x_center / self.image_width) - 1  # [-1, 1]
                y_norm = 2 * (y_center / self.image_height) - 1  # [-1, 1]
                bounding_boxes[idx] = [w, h, x_norm, y_norm]
            else:
                bounding_boxes[idx] = [0, 0, 0, 0]

        # Assemble identities
        identities = np.full(num_modified, -1, dtype=int)
        for idx, cond_idx in enumerate(modified_indices):
            if self.identities[cond_idx] is not None:
                identities[idx] = self.identities[cond_idx]
            else:
                identities[idx] = 0  # or -1 to indicate "no identity"

        # Assemble source masks
        source_masks = np.array(
            [self.source_masks[self.source_frame_indices[cond_idx]] for cond_idx in modified_indices]
        )

        # Assemble target masks
        target_masks = np.array([self.target_masks[cond_idx] for cond_idx in modified_indices])

        # Assemble source indices
        source_indices = np.array(
            [self.source_frame_indices[cond_idx] for cond_idx in modified_indices], dtype=int
        )

        # Assemble target indices
        target_indices = np.array(
            [self.target_frame_indices[cond_idx] for cond_idx in modified_indices], dtype=int
        )

        # Collect ALL source images (similarly done in the original code)
        source_images_list = []
        for idx in range(len(self.source_frames)):
            img = self.source_frames[idx]
            source_images_list.append(img)
        source_images = np.array(source_images_list)  # (n_source_frames, H, W, C)
        source_images = np.transpose(source_images, (0, 3, 1, 2))  # (n_source_frames, C, H, W)

        # Collect ALL target images (this is the new part!)
        target_images_list = []
        for idx in range(len(self.target_frames)):
            img = self.target_frames[idx]
            target_images_list.append(img)
        target_images = np.array(target_images_list)  # (n_target_frames, H, W, C)
        target_images = np.transpose(target_images, (0, 3, 1, 2))  # (n_target_frames, C, H, W)

        # Assemble trajectories
        trajectories = []
        for cond_idx in modified_indices:
            if self.trajectories[cond_idx] is not None:
                x_traj, y_traj = self.trajectories[cond_idx]
                traj = np.vstack((x_traj, y_traj)).T  # (n_points, 2)
                trajectories.append(traj)
            else:
                trajectories.append(None)

        # Condition numbers
        condition_numbers = np.array(modified_indices, dtype=int)

        return {
            "source_images": source_images,
            "target_images": target_images,  # <--- NEWLY ADDED
            "bounding_boxes": bounding_boxes,
            "identities": identities,
            "source_masks": source_masks,
            "target_masks": target_masks,
            "source_indices": source_indices,
            "target_indices": target_indices,
            "condition_numbers": condition_numbers,
            "trajectories": trajectories,
        }

    def update_trajectory_window(self, x, y):
        """
        Adjust the trajectory window and display a cropped version of the source frame.
        """
        self.x_min = max(0, x - 50)
        self.x_max = min(self.image_width, x + 50)
        self.y_min = max(0, y - 100)
        self.y_max = min(self.image_height, y + 100)

        source_frame_index = self.source_frame_indices[self.current_condition]
        img = self.source_frames[source_frame_index]
        img_cropped = img[int(self.y_min) : int(self.y_max), int(self.x_min) : int(self.x_max), :]

        self.ax_trajectory.clear()
        self.ax_trajectory.imshow(img_cropped)
        self.ax_trajectory.set_xlim(0, self.x_max - self.x_min)
        self.ax_trajectory.set_ylim(self.y_max - self.y_min, 0)  # Invert y-axis
        self.ax_trajectory.set_title("Trajectory")

        # Re-initialize the trajectory line
        (self.trajectory_line,) = self.ax_trajectory.plot([], [], "b-")

        # If a trajectory exists for this condition, redraw it
        if self.trajectories[self.current_condition] is not None:
            x_traj, y_traj = self.trajectories[self.current_condition]
            x_traj_adj = [xi - self.x_min for xi in x_traj]
            y_traj_adj = [yi - self.y_min for yi in y_traj]
            self.ax_trajectory.plot(x_traj_adj, y_traj_adj, "b-")

        self.fig.canvas.draw_idle()

    def on_button_press(self, event):
        """
        Handle mouse button press for both the trajectory drawing and tile clicks.
        """
        if event.inaxes == self.ax_trajectory and self.current_mode == "trajectory":
            self.drawing = True
            x = event.xdata + self.x_min
            y = event.ydata + self.y_min
            self.trajectory_x = [x]
            self.trajectory_y = [y]
            x_disp = [x - self.x_min]
            y_disp = [y - self.y_min]
            self.trajectory_line.set_data(x_disp, y_disp)
            self.fig.canvas.draw_idle()
        else:
            # Otherwise, handle tile click
            self.on_click(event)

    def on_button_release(self, event):
        """
        Handle mouse button release for trajectory drawing.
        """
        if event.inaxes == self.ax_trajectory and self.current_mode == "trajectory":
            self.drawing = False
            self.trajectories[self.current_condition] = (
                self.trajectory_x.copy(),
                self.trajectory_y.copy(),
            )
            self.modified_conditions.add(self.current_condition)

    def on_motion(self, event):
        """
        Handle mouse movement while drawing a trajectory.
        """
        if self.drawing and event.inaxes == self.ax_trajectory and self.current_mode == "trajectory":
            x = event.xdata + self.x_min
            y = event.ydata + self.y_min
            self.trajectory_x.append(x)
            self.trajectory_y.append(y)

            x_disp = [xi - self.x_min for xi in self.trajectory_x]
            y_disp = [yi - self.y_min for yi in self.trajectory_y]
            self.trajectory_line.set_data(x_disp, y_disp)
            self.fig.canvas.draw_idle()

    def load_trajectory(self):
        """
        Reload the trajectory plot for the current condition.
        """
        self.ax_trajectory.clear()
        (self.trajectory_line,) = self.ax_trajectory.plot([], [], "b-")
        self.ax_trajectory.set_title("Trajectory")

        if self.trajectories[self.current_condition] is not None:
            x_traj, y_traj = self.trajectories[self.current_condition]
            x_traj_adj = [xi - self.x_min for xi in x_traj]
            y_traj_adj = [yi - self.y_min for yi in y_traj]
            self.ax_trajectory.plot(x_traj_adj, y_traj_adj, "b-")

        self.ax_trajectory.set_xlim(0, self.image_width)
        self.ax_trajectory.set_ylim(self.image_height, 0)  # Invert y-axis
        self.fig.canvas.draw_idle()


if __name__ == "__main__":
    # Example usage

    filename = "actions/1/action1.pkl"
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            loaded_data = pickle.load(f)
        print(f"Loaded data from {filename}")
    else:
        loaded_data = None
        print("No existing data file found. Starting fresh.")

    # Your source and target image paths
    source_frame_paths = [
        "examples_high_quality/UDBssfvV95E_6625_5.98_46.02.jpg",
        "examples_high_quality/qCts0RXfl8w_11800_5.54_54.39.jpg",
        "demo_samples/10/_9x-f3sehns_8175_4.95_63.45.jpg",
    ]
    target_frame_paths = [
        "examples_high_quality/UDBssfvV95E_6625_5.98_46.02.jpg",
    ]

    tool = SelectionTool(
        source_frame_paths=source_frame_paths,
        target_frame_paths=target_frame_paths,
        resized_height=576,
        resized_width=1024,
        grid_height=576 // 16,
        grid_width=1024 // 16,
        n_target_frames=25,
        n_conditions=5,
        max_num_identities=5,
        loaded_data=None, #loaded_data,
    )
    data = tool.run()

    # Unpack results
    source_images = data["source_images"]
    target_images = data["target_images"]  # <--- NOW ALSO AVAILABLE
    bounding_boxes = data["bounding_boxes"]
    identities = data["identities"]
    source_masks = data["source_masks"]
    target_masks = data["target_masks"]
    source_indices = data["source_indices"]
    target_indices = data["target_indices"]
    condition_numbers = data["condition_numbers"]
    trajectories = data["trajectories"]

    print("Modified Conditions:", condition_numbers)
    print("Source Images Shape:", source_images.shape)
    print("Target Images Shape:", target_images.shape)  # <--- CHECK THIS
    print("Bounding Boxes Shape:", bounding_boxes.shape)
    print("Identities Shape:", identities.shape)
    print("Source Masks Shape:", source_masks.shape)
    print("Target Masks Shape:", target_masks.shape)
    print("Source Indices Shape:", source_indices.shape)
    print("Target Indices Shape:", target_indices.shape)
    print("Number of Trajectories:", len(trajectories))

    # Save to file
    while os.path.exists(filename):
        dir_num = filename.split("/")[1]
        file_num = filename.split(".")[0].split("/")[-1].split("action")[-1]
        file_num = int(file_num)
        filename = f"actions/{dir_num}/action{file_num+1}.pkl"
        print(f"Saving data to {filename}")
    with open(filename, "wb") as f:
        pickle.dump(data, f)
    print(f"Data saved to {filename}")

    # Reload and verify
    with open(filename, "rb") as f:
        loaded_data = pickle.load(f)

    # Check that everything matches
    assert np.array_equal(source_images, loaded_data["source_images"])
    assert np.array_equal(target_images, loaded_data["target_images"])  # <--- VERIFY TARGET IMAGES
    assert np.array_equal(bounding_boxes, loaded_data["bounding_boxes"])
    assert np.array_equal(identities, loaded_data["identities"])
    assert np.array_equal(source_masks, loaded_data["source_masks"])
    assert np.array_equal(target_masks, loaded_data["target_masks"])
    assert np.array_equal(source_indices, loaded_data["source_indices"])
    assert np.array_equal(target_indices, loaded_data["target_indices"])
    assert np.array_equal(condition_numbers, loaded_data["condition_numbers"])

    for (t1, t2) in zip(trajectories, loaded_data["trajectories"]):
        if t1 is None and t2 is None:
            continue
        elif (t1 is None) ^ (t2 is None):
            raise ValueError("Trajectory mismatch: one is None, the other is not.")
        else:
            # Compare each trajectory's contents
            if not np.array_equal(t1, t2):
                raise ValueError("Trajectory data mismatch.")

    print("All loaded data matches the original (including target_images).")