from pydantic import BaseModel, RootModel


class ParamsModel(BaseModel):
    value: float | str
    unit: str
    source: str = ""
    long_name: str = ""


class MaterialModel(BaseModel):
    material_class: str
    elements: dict[str, float]
    density: float
    poisson_ratio: float


class MassFractionMaterialModel(BaseModel): ...


class MixtureModel(BaseModel):
    materials: dict[str, float]


class MaterialsModel(BaseModel):
    materials: str | MaterialModel | None = None
    mixtures: str | MixtureModel | None = None


class DesignerConfig(BaseModel):
    params: ParamsModel | None = None
    run_mode: str = "run"


class BuilderConfig(BaseModel):
    params: ParamsModel | None = None
    material: dict[str, str] | str | None = None


class BuildStageConfig(BaseModel):
    designer: DesignerConfig | None = None
    builder: BuilderConfig | None = None


class BuildConfig(RootModel):
    params: ParamsModel | None = None
    materials_path: MaterialsModel | None = None
    root: dict[str, DesignerConfig | BuilderConfig | BuildStageConfig] | None = None

    def __getitem__(self, item):
        return self.root[item]
