import numpy as np

def isOverlapping1D(box1, box2):
    return box1[1] >= box2[0] and box1[0] <= box2[1]

def isOverlapping2D(box1, box2):
    return isOverlapping1D([box1[0], box1[2]], [box2[0], box2[2]]) and isOverlapping1D([box1[1], box1[3]], [box2[1], box2[3]])

def isOverlappingTriplet(pred_triplet):
    """
    Return True if the subject and object bounding boxes overlap.

    Args:
        pred_triplet: [num_rel, 8] array with the following columns:
            0:4 - subject bounding box
            4:8 - object bounding box

    Returns:
        [num_rel] array of booleans
    """

    return np.array([isOverlapping2D(pred_triplet[i, :4], pred_triplet[i, 4:]) for i in range(pred_triplet.shape[0])])

def filterSubjectRelations(rel_scores, K):
    """
    Return the K most likely triplet predictions where subject is fixed.

    Args:
        rel_scores: [num_rel, 53] array of relation scores where the last 51 columns are the relation scores

    Returns:
        Filtered relation indices
    """

    num_rel = rel_scores.shape[0]
    tripletScores = rel_scores[:, 2:].reshape(-1, 51)
    topK = np.argsort(tripletScores, axis=0)[::-1][:, tripletScores.shape[1] - K:tripletScores.shape[1]]

    return topK


def filterPhysicalConstratins(pred_triplets, idx_to_label, ind_to_predicate):
    """
    Based on the physical constraints, remove the triplets that are not valid.

    Args:
        pred_triplets: [num_triplets, 3] (subject_class, relation, object_class) array

    Returns:
        Filtered triplets
    """

    tags = {1: "street", # roads
            2: "sidewalk", # sidewalks
            3: "building", # buildings
            6: "pole",  # poles
            12: "person", # pedestrians
            14: "car",  # cars 
            15: "truck", # trucks
            16: "bus", # buses
            18: "motorcycle", # motorcycles
            19: "bike", # bicycles
            24: "sign"} # RoadLines (not sure)
                        # The tags are from the CARLA ObjectLabels.h

    toKeep = []
    for i, triplet in enumerate(pred_triplets):
        subject = idx_to_label[str(int(triplet[0]))]
        predicate = ind_to_predicate[str(int(triplet[1]))]
        obj = idx_to_label[str(int(triplet[2]))]

        subjectFilterResult = False
        PhyicsFilterResult = True
        # (subject, predicate) filtering
        for cls in tags.values():
            if cls != subject:
                continue    

            if cls == "building":
                if predicate in ["next to", "in front of", "behind", "across", "between", "has", "looking at", "made of", "near", "under"]:
                    subjectFilterResult = True
                break

            if cls == "street":
                if predicate in ["next to", "in front of", "behind", "across", "between", "has", "near", "under"]:
                    subjectFilterResult = True
                break
            if cls == "sidewalk":
                if predicate in ["next to", "in front of", "behind", "across", "between", "has", "near", "under", "to", "with", "part of", "on"]:
                    subjectFilterResult = True
                break
            if cls in ["car", "truck", "bus", "motorcycle", "bike"]:
                if predicate in ["above", "and", "belonging to", "carrying", "covered in", "parked on", "riding", "next to", "in front of", "behind", "across", "between", "has", "near", "under", "to", "with", "part of"]:
                    subjectFilterResult = True
                break

            if cls == "person":
                if predicate in ["above", "and", "carrying", "covered in", 
                                 "walking on", "next to", "in front of", "behind", "across", 
                                 "between", "has", "near", "under", "to", "with", "part of", "on"]:
                    subjectFilterResult = True
                break
                        
            if cls == "pole":
                if predicate in ["above", "and", "carrying", "covered in",
                                  "next to", "in front of", "behind", "across", "between", "has", "near", 
                                  "under", "to", "with", "part of", "on"]:
                    subjectFilterResult = True
                break
            if cls == "sign":
                if predicate in ["next to", "in front of", "behind", "across", "between", "has", "near", "under"]:
                    subjectFilterResult = True
                break

        # Common Sense filtering (remove building -> on -> car type of relations)
        PhyicsFilterResult = is_realistic_triplet(subject, predicate, obj)

        if subjectFilterResult and PhyicsFilterResult:
            toKeep.append(i)

def is_realistic_triplet(subject, relation, obj):
    # Rule 1: "Sign" (road lines) cannot have spatial relations like "on", "above", etc.
    if subject == "sign" or obj == "sign":
        if relation in {"on", "above", "under", "over", "in front of", "behind", "mounted on"}:
            return False

    # Rule 2: "Person" can only "ride", "walk on", "stand on", or "sit on" a vehicle, not the other way around.
    if subject in {"car", "truck", "bus", "motorcycle", "bike"} and obj == "person":
        if relation in {"riding", "walking on", "standing on", "sitting on"}:
            return False

    # Rule 3: "Building" or "pole" cannot be "on" any other object.
    if subject in {"building", "pole"} and relation == "on":
        return False

    # Rule 4: Vehicles cannot be "on" a person, building, or pole.
    if subject in {"car", "truck", "bus", "motorcycle", "bike"} and obj in {"person", "building", "pole"}:
        if relation == "on":
            return False

    # Rule 5: "Street" or "sidewalk" can only have objects "on" or "along" them, not the other way around.
    if subject in {"street", "sidewalk"} and relation not in {"on", "along"}:
        return False
    if obj in {"street", "sidewalk"} and relation in {"on", "along"}:
        return False

    # Rule 6: Vehicles cannot "carry" or "wear" objects.
    if subject in {"car", "truck", "bus", "motorcycle", "bike"} and relation in {"carrying", "wearing", "wears"}:
        return False

    # Rule 7: "Pole" cannot "hold" or "carry" anything.
    if subject == "pole" and relation in {"holding", "carrying"}:
        return False

    # Rule 8: Vehicles can only "park on" or "ride on" streets or sidewalks.
    if subject in {"car", "truck", "bus", "motorcycle", "bike"}:
        if obj not in {"street", "sidewalk"} and relation in {"parked on", "riding"}:
            return False

    # Rule 9: People cannot "paint on" or "mounted on" vehicles or poles.
    if subject == "person" and relation in {"painted on", "mounted on"} and obj in {"car", "truck", "motorcycle", "bike", "pole"}:
        return False

    # Rule 10: "Person" can "hold" or "wear" objects but not "street", "sidewalk", "pole", or vehicles.
    if subject == "person" and relation in {"holding", "wearing", "wears"}:
        if obj in {"street", "sidewalk", "pole", "car", "truck", "bus", "motorcycle", "bike"}:
            return False

    return True
