from kg_gen.models import Graph
from typing import List, Tuple
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import dspy



class ClassificationRelation(BaseModel):
    """Knowledge graph entity-classification tuple."""

    entity: str = dspy.InputField(desc="Entity", examples=["SampleLibrary"])
    classification: str = dspy.InputField(desc="Classification or NONE", examples=["Component", "Connector", "NONE"])
    is_none: str = dspy.InputField(desc="Is classification NONE", examples=["False", "True"])

def get_classification_sig(ontology: str) -> dspy.Signature:
    class ClassifyEntities(dspy.Signature):
        __doc__ = f"""Classify key entities from the source text according to the following ontology if possible. 
        Classify according to their type, based on the ONTOLOGY and the list of ontology classes provided. Provide NONE if no type matches.
        BEGIN ONTOLOGY:
        {ontology}
        END ONTOLOGY.
        Please be THOROUGH and accurate to the reference text."""

        source_text: str = dspy.InputField(desc="Source text containing the entities to classify")
        entities: list[str] = dspy.InputField(desc="THOROUGH list of already identified key entities")
        ontology_classes: list[str] = dspy.InputField(desc="List of ontology classes to classify entities into", examples=["Component", "Connector", "Property"])
        classifications: list[ClassificationRelation] = dspy.OutputField(
                desc="List of entity-classification tuples where entity is an exact match to items in entities list. Classification is a single classfication according to the provided ONTOLOGY or NONE. Be thorough"
            )
        
    return ClassifyEntities
    
    
def classify_entities(
    input_data: str,
    entities: list[str],
    ontology_definition: str,
    ontology_classes: List[str],
) -> List[Tuple[str, str]]:
    classification_sig = get_classification_sig(ontology_definition)
    
    try:
        extract = dspy.Predict(classification_sig)
        result = extract(source_text=input_data, entities=entities, ontology_classes=ontology_classes)
    except Exception as e:
        print("Got exception during classification")
        return []
    
    classifications: list[ClassificationRelation] = result.classifications
    return [(r.entity, r.classification) for r in classifications]