from pydantic import BaseModel
from typing import List

class Input(BaseModel):
    age: int
    sex: str
    on_thyroxine: str
    query_on_thyroxine: str
    on_antithyroid_medication: str
    sick: str
    pregnant: str
    thyroid_surgery: str
    I131_treatment: str
    query_hypothyroid: str
    query_hyperthyroid: str
    lithium: str
    goitre: str
    tumor: str
    hypopituitary:str
    psych: str
    TSH_measured: str
    TSH: float
    T3_measured: str
    T3: float
    TT4_measured: str
    TT4: float
    T4U_measured: str
    T4U: float
    FTI_measured: str
    FTI: float
    TBG_measured: str
    TBG: float
    referral_source: str
