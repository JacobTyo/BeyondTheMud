import wandb

class WandbWriter:
    """Writes entries directly to wandb.

    This class provides a high-level API to log metrics, images, and other
    data types to wandb, which can be used to visualize and compare runs.
    """

    def __init__(
        self,
        project_name,
        config=None,
    ):
        """Initializes WandbWriter to log data to wandb.

        Args:
            project_name (str): Name of the wandb project.
            config (dict, optional): A dictionary containing configuration parameters for the run.
        """
        wandb.init(project=project_name, config=config)
        self.step = 0

    def set_step(self, step):
        """Sets the global step value for logging.

        Args:
            step (int): The global step value.
        """
        self.step = step

    def log(self, data, step=None):
        """Logs the given data to wandb.

        Args:
            data (dict): A dictionary where keys are the names of the metrics or data points, and values are the actual data.
        """
        if not step:
            step = self.step
            self.step += 1
        wandb.log(data, step=step)

    def log_image(self, tag, image, step=None):
        """Logs an image to wandb.

        Args:
            tag (str): Name of the image.
            image (PIL.Image or numpy.ndarray): Image data.
            step (int, optional): Step value to associate with the image. If None, uses the internal step value.
        """
        step = step if step is not None else self.step
        wandb.log({tag: wandb.Image(image)}, step=step)

    def add_scalar(self, tag, value, step):
        self.log({tag: value})

    def close(self):
        """Closes the wandb run."""
        wandb.finish()
