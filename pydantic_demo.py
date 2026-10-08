from pydantic import BaseModel, ConfigDict


class PetCls:
    def __init__(self, *, name: str) -> None:
        self.name = name


class PersonCls:
    def __init__(self, *, name: str, pets: list[PetCls]) -> None:
        self.name = name
        self.pets = pets


class Pet(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    name: str


class Person(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    name: str
    pets: list[Pet]


bones = PetCls(name='Bones')
orion = PetCls(name='Orion')
anna = PersonCls(name='Anna', pets=[bones, orion])
anna_model = Person.model_validate(anna)
print(anna_model)
