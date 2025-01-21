from typing import Any

from pydantic import BaseModel, RootModel


class ParamsModel(BaseModel):
    value: float | str
    unit: str
    source: str = ""
    long_name: str = ""
    description: str = ""


class MaterialModel(BaseModel):
    material_class: str
    elements: dict[str, float]
    density: float
    poisson_ratio: float


class MassFractionMaterialModel(BaseModel): ...


class MixtureModel(BaseModel):
    materials: dict[str, float]


class MaterialsModel(BaseModel):
    materials: str | dict[str, MaterialModel] | None = None
    mixtures: str | dict[str, MixtureModel] | None = None


class DesignerConfig(BaseModel):
    params: dict[str, ParamsModel] | str | None = None
    run_mode: str = "run"
    extra_keys: dict[str, Any] = {}

    class Config:
        extra = "allow"


class BuilderConfig(BaseModel):
    params: dict[str, ParamsModel] | str | None = None
    material: dict[str, str] | str | None = None
    extra_keys: dict[str, Any] = {}

    class Config:
        extra = "allow"


class BuildStageConfig(BaseModel):
    designer: DesignerConfig | None = None
    builder: BuilderConfig | None = None


class BuildConfig(RootModel):
    root: dict[
        str,
        dict[str, ParamsModel]
        | str
        | MaterialsModel
        | DesignerConfig
        | BuilderConfig
        | BuildStageConfig,
    ]

    def __getitem__(self, item):
        return self.root[item]

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            try:
                item = item.replace("_", " ")
                return self[item]
            except KeyError:
                pass
        return super().__getattr__(self, item)
