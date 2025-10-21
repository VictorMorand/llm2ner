from typing import (
    Dict,
    Type,
    TypeVar,
    Union,
    Optional,
)
import torch, os, json, logging, shutil
import torch.nn as nn
from experimaestro.core.arguments import Meta
from experimaestro import (
    Config,
    Task,
    LightweightTask,
    Param,
    Constant,
    DataPath,
    serialize,
    deserialize,
)
from experimaestro.core.context import SerializedPath
from pathlib import Path
from huggingface_hub import ModelHubMixin, snapshot_download, hf_hub_download


T = TypeVar("T", bound="xpmTorchHubModule")


class xpmTorchHubModule(nn.Module, Config, ModelHubMixin):
    """
    Generic PyTorch module for experimaestro, 
    compatible with Hugging Face Hub via [ModelHubMixin](https://huggingface.co/docs/huggingface_hub/en/package_reference/mixins).

    Example usage:
    ```python
    class MyTorchxpmTorchHubModule(xpmTorchHubModule,
        library_name="my-org/my-model",
        tags=["example", "torch"],
        repo_url="https://huggingface.co/my-org/my-model",
        paper_url="https://arxiv.org/abs/???"
        ):

        ## Child parameters
        input_dim: Param[int] = 100

        hidden_dim: Param[int] = 200

        output_dim: Param[int] = 10

        def __post_init__(self):
            # Here we use the child parameters to initialize our module layers and other stuff
            super().__post_init__()
            self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
            self.fc2 = nn.Linear(self.hidden_dim, self.output_dim)
            logging.debug("Initialized layers: %s, %s", self.fc1, self.fc2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    #new model configuration
    model_cfg:Config = MyTorchxpmTorchHubModule.C(input_dim=50, hidden_dim=100, output_dim=5)

    #instanciate the model fomr config
    model = model_cfg.instance()

    # Saving
    #save to local directory
    model.save_pretrained("my_model_dir")
    #push to HF hub
    model.push_to_hub("my_model_id")

    # Loading
    #load from file
    new_model = MyTorchxpmTorchHubModule.from_pretrained("my_model_dir")
    #load from HF hub
    new_model = MyTorchxpmTorchHubModule.from_pretrained("my_model", force_download=True)
    ```
    """

    class Loader(LightweightTask):
        """Loader for the xpmTorchHubModule"""

        model: Param["xpmTorchHubModule"]
        parameters: Meta[DataPath]

        def execute(self):
            """Loads the model from disk using the given serialization path"""
            state_dict = torch.load(self.parameters, map_location="cpu")
            self.model.load_state_dict(state_dict)

    def __post_init__(self):
        """Initialize the module. Child classes should override this method to initialize their layers and other stuff."""
        super().__post_init__()
        

    @property
    def device(self):
        return next(self.parameters()).device

    def count_parameters(self):
        """Count the number of parameters in the model"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # def save_pretrained(
    #     self,
    #     save_directory,
    #     *,
    #     config=None,
    #     repo_id=None,
    #     push_to_hub=False,
    #     model_card_kwargs=None,
    #     **push_to_hub_kwargs,
    # ):
    #     """ Save the model weights and config to the specified directory.
    #     see <https://huggingface.co/docs/huggingface_hub/en/package_reference/mixins#huggingface_hub.ModelHubMixin>
    #     """

    #     return super().save_pretrained(
    #         save_directory,
    #         config=config,
    #         repo_id=repo_id,
    #         push_to_hub=push_to_hub,
    #         model_card_kwargs=model_card_kwargs,
    #         **push_to_hub_kwargs,
    #     )

    def _save_pretrained(self, save_directory: Path):
        """
        Save the model weights and config to the specified directory.
        """

        # Save model weights to directory
        params_path = Path("parameters.pth")
        torch.save(self.state_dict(), params_path)
        if save_directory.exists():
            logging.warning(f"Folder {save_directory} already exists, overwriting it")
            shutil.rmtree(save_directory)
        # serialize whole model and dependancies
        xpm_config = self.__config__
        serialize(
            xpm_config,
            save_directory,
            init_tasks=[self.Loader.C(model=xpm_config, parameters=params_path)],
        )

        # # finally delete temporary files
        # params_path.unlink(missing_ok=True)

    @classmethod
    def _from_pretrained(
        cls: Type[T],
        *,
        model_id: str,
        revision: Optional[str],
        cache_dir: Optional[Union[str, Path]],
        force_download: bool,
        proxies: Optional[Dict],
        resume_download: Optional[bool],
        local_files_only: bool,
        token: Optional[Union[str, bool]],
        **model_kwargs,
    ) -> T:
        """
        Load the model weights and config from the specified directory.
        Rest of loading logic is done by HF ModelHubMixin.
        see <https://huggingface.co/docs/huggingface_hub/en/package_reference/mixins#huggingface_hub.ModelHubMixin>
        """
        if os.path.isdir(model_id):
            #model is save Locally
            save_directory = Path(model_id)
            model_cfg, [loader] = deserialize(save_directory, as_instance=False)
            #get Loader Task
            loader = loader.instance()
            loader.execute()
            model = loader.model
            object.__setattr__(model, "__config__", model_cfg)
            return model

        else:

            def data_loader(s_path: Union[Path, str, SerializedPath]):
                if not isinstance(s_path, SerializedPath):
                    s_path = SerializedPath(Path(s_path), False)
                path = s_path.path

                # Folder
                if s_path.is_folder:
                    hf_path = snapshot_download(
                        repo_id=model_id,
                        allow_patterns=f"{s_path.path}/**",
                        revision=revision,
                        cache_dir=cache_dir,
                        proxies=proxies,
                        resume_download=resume_download,
                        token=token,
                        local_files_only=local_files_only,
                    )
                    return Path(hf_path) / path

                hf_path = Path(
                    hf_hub_download(
                        repo_id=model_id,
                        filename=str(path),
                        revision=revision,
                        cache_dir=cache_dir,
                        force_download=force_download,
                        proxies=proxies,
                        resume_download=resume_download,
                        token=token,
                        local_files_only=local_files_only,
                    )
                )
                return hf_path

        return deserialize(data_loader, as_instance=True)

    @classmethod
    def from_kwargs(cls, **kwargs) -> "xpmTorchHubModule":
        """
        Alternate constructor that parses kwargs into xpmConfig and returns an instance.
        """
        cfg = cls.C(**kwargs)
        return cfg.instance()




if __name__ == "__main__":

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)


    # Example child module
    class MyTorchxpmTorchHubModule(
        xpmTorchHubModule,
        library_name="my-org/my-model",
        tags=["torch", "experimaestro"],
        repo_url="https://github.com/VictorMorand/test-model",
        paper_url="https://arxiv.org/abs/???",
    ):
        """Example xpmTorchHubModule implementation, now all saving and loading is taken care of under the hood"""

        ## Child parameters
        input_dim: Param[int] = 100
        """Input dimension"""

        hidden_dim: Param[int] = 200
        """Hidden dimension"""

        output_dim: Param[int] = 10
        """Output dimension"""

        version: Constant[str] = "1.0"

        def __post_init__(self):
            super().__post_init__()
            # self._config = self.__config__

            self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
            self.fc2 = nn.Linear(self.hidden_dim, self.output_dim)
            logging.debug("Initialized layers: %s, %s", self.fc1, self.fc2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x


    model_id = "VictorMorand/test-model"
    save_path = Path("./test_model")

    create_new_model = False
    create_new_model = True

    if create_new_model:
        # create configuration
        cfg = MyTorchxpmTorchHubModule.C(input_dim=50, hidden_dim=100, output_dim=5)
        # model = xpmTorchHubModule.from_kwargs(input_dim=50, hidden_dim=100, output_dim=5)
        # model = xpmTorchHubModule(input_dim=50, hidden_dim=100, output_dim=5)

        model: MyTorchxpmTorchHubModule = cfg.instance()
        print(model)
        xpm_config = model.__config__
        print(type(xpm_config),xpm_config)
        
        # test it
        input = torch.randn(1, 50)
        output = model(input)

        print(output)

        model.save_pretrained(save_path)
        # model.push_to_hub(model_id)

        # try to reload it
        logging.info("Reloading model from %s", save_path)
        newModel = MyTorchxpmTorchHubModule.from_pretrained(save_path)

    else:
        # save_path = Path("./test_model")
        # try to load from Hub

        newModel = MyTorchxpmTorchHubModule.from_pretrained(
            model_id, force_download=True, local_files_only=False
        )

    logging.info("New model: %s", newModel)
    logging.info("New model config: %s", newModel.__config__)

    # # # Model that can be re-used in experiments
    # # model, init_tasks = AutoModel.load_from_hf_hub("xpmir/SPLADE_DistilMSE")

    # # init_tasks[0].execute()
    # # Use this if you want to actually use the model
    # model = AutoModel.load_from_hf_hub("xpmir/SPLADE_DistilMSE", as_instance=True)
    # print(model.rsv("walgreens store sales average", "The average Walgreens salary ranges..."))
