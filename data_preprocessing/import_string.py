import pandas as pd

def remove_redun(el, verbose=False):
    if verbose:
        print("Original Size: ", len(el))

    el_new = el.iloc[:, 0:2].apply(sorted, axis=1)

    el_new = pd.DataFrame.from_dict(dict(zip(el_new.index, el_new.values))).T 

    el_new = el_new.drop_duplicates()
    if verbose:
        postDrop = len(el_new)
        print("After Dropping Duplicates: ", len(el_new), "(-", len(el)-postDrop, ")")

    el_new = el_new.merge(el, left_on=[el_new.columns[0],el_new.columns[1]],  right_on=[el.columns[0], el.columns[1]])
    if verbose:
        print("After Merging: ", len(el_new), "(-", postDrop-len(el_new), ")")
        print()

    return el_new.iloc[:, 2:]

def map_IDs(el, gmap, verbose = False, dropNaNvalues = True):
    gp_map_f = gmap.set_index('#string_protein_id')['alias']

    el_converted = el.reset_index(drop=True)

    el_converted[el_converted.columns[0]] = el_converted[el_converted.columns[0]].map(gp_map_f)
    el_converted[el_converted.columns[1]] = el_converted[el_converted.columns[1]].map(gp_map_f)

    if verbose:
        print("NaN values per Column:", el_converted[el_converted.columns[0]].isna().sum(), el_converted[el_converted.columns[1]].isna().sum())

    if dropNaNvalues:
        el_converted = el_converted.dropna()
        if verbose:
            print("New edge list size:", len(el_converted), "( -", len(el)-len(el_converted), ")")
    
    return el_converted

def ImportSTRING():
    el_map = pd.read_csv('../data/raw_data/9606.protein.aliases.v11.5.txt', sep="\t")
    el = pd.read_csv('../data/raw_data/9606.protein.links.detailed.v11.5.txt', sep=" ")
    el_map = el_map.loc[el_map.source == 'Ensembl_gene']

    el = remove_redun(el, True)
    el = map_IDs(el, el_map, verbose=True)

    return el