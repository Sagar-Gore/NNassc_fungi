#!/usr/bin/env python

### NNassc - Neural Network based Adenylation domain substrate specificity classifier ####

### Prediction of substrates- with nine NRPS code residues as an input 

'''GNU GENERAL PUBLIC LICENSE
                       Version 3, 29 June 2007

 Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>
 Everyone is permitted to copy and distribute verbatim copies
 of this license document, but changing it is not allowed.

 ## Please see copy of LICENSE for further details ###'''


import argparse 
parser = argparse.ArgumentParser(usage ='python3.7 NNassc_final.py --input NRPScode_9residues', 
    description= 'Input: 9 NRPScode residues Output: Substrates')

parser.add_argument('--input', type=str, required=True, help="9 NRPS code residues")

args = parser.parse_args()
if len(args.input) == 9:
    aacode = args.input
else:
    print ("please provide the correct input - 9 NRPS code residues")


def run_NNassc(aacode):
    import pickle, sys
    def NRPS_training_data():
        from sklearn.tree import export_graphviz
        import numpy as np
        FP_size_Amino = "4096"
        FP_size_Ligand = int(1024)

        def dict_normalize(input_dict):
        ### normalization of python dictionary ###
            from sklearn import preprocessing
            out_dict = {}
            vals = list([list(input_dict.values())])
            norm_vals = preprocessing.normalize(vals, norm='l1')
            valsNorm = norm_vals[0]
            cnt = 0
            for x in input_dict.keys():
                if x not in out_dict:
                    out_dict [x] = round(valsNorm[cnt], 3)
                    cnt = cnt + 1  
            return (out_dict)

        def get_aaindex(amino):
        ### map Wold encoding aaindex for amino acid ###
            import aaindex
            import numpy as np
            list_ind = []
            aaindex.init(path='.')
            z1 = aaindex.get('WOLS870101')
            z2 = aaindex.get('WOLS870102')
            z3 = aaindex.get('WOLS870103')
            Z1_norm =  dict_normalize(z1.index)
            Z2_norm =  dict_normalize(z2.index)
            Z3_norm =  dict_normalize(z3.index)
            sze = Z1_norm[amino] 
            shp = Z2_norm[amino]
            ele = Z3_norm[amino]
            list_ind.append(sze)
            list_ind.append(shp)
            list_ind.append(ele)
            return (list_ind)

        def SMILES2bitvect(SMILE, FP_size_Ligand):
        ## Conversion of SMILES strings into a Morgan fingerperint bitvector ###
            from rdkit import Chem 
            from rdkit.Chem import AllChem
            mol = Chem.MolFromSmiles(SMILE)
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=FP_size_Ligand, useFeatures=True)
            fp_bitstr = fp.ToBitString()
            fp_bitstr_list = [int(P) for P in fp_bitstr]
            return (fp_bitstr_list) 

        def SMILES2bitvect_prot(SMILE, FP_size_Amino):
        ### Amino acid SMILES strings to MorganFingerprint bitvector ###   
            from rdkit import Chem 
            from rdkit.Chem import AllChem    
            mol = Chem.MolFromSmiles(SMILE)
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=FP_size_Amino, useFeatures=True)
            fp_bitstr = fp.ToBitString()
            fp_bitstr_list = [int(P) for P in fp_bitstr]
            return (fp_bitstr_list) 

        def aminoMorgan_to30bits(SMILE, FP_size_Amino):
        ### Transforming amino acid 4096-bits bitvector to 30-bits bitvector ###
            import numpy as np
            oldbitvect= SMILES2bitvect_prot(SMILE, FP_size_Amino)
            ONbits_amino = ["0" ,"1" ,"19" ,"2" ,"3" ,"32" ,"3133" ,"2840" ,"3000" ,"1668" ,"3656" ,"511" ,"2961" ,"1409" 
                        ,"3702" ,"1760" ,"2752" ,"2370" ,"819" ,"2087" ,"1033" ,"3999" ,"2853" ,"2397" ,"901" ,"3096" ,"3998" ,"4089" ,"1649" ,"1013"]
            new_30bitvect = []
            for i in ONbits_amino:
                new_30bitvect.append(oldbitvect[int(i)])
            return new_30bitvect

        def convert01_minus1plus1(btvct1):
        ### transform binary encoding to -1 & 1 (0 ---> -1 & 1 ---> 1) ###
            outbitvct1 = []
            for ibit in btvct1:
                if ibit == 1:
                    outbitvct1.append(int(1))
                else:
                    outbitvct1.append(int(-1))
            return (outbitvct1)

        def amino_to_SMILES(aminoacid):
        ### amino acid single letter code to canonical SMILES strings ### 
            amino_SMILE = {"A":"CC(C(=O)O)N", "C":"C(C(C(=O)O)N)SSCC(C(=O)O)N", "D":"C(C(C(=O)O)N)C(=O)O", "E":"C(CC(=O)O)C(C(=O)O)N", "F":"C1=CC=C(C=C1)CC(C(=O)O)N", "G":"C(C(=O)O)N", "H":"C1=C(NC=N1)CC(C(=O)O)N", 
                   "I":"CCC(C)C(C(=O)O)N", "K":"C(CCN)CC(C(=O)O)N", "L":"CC(C)CC(C(=O)O)N", "M":"CSCCC(C(=O)O)N", "N":"C(C(C(=O)O)N)C(=O)N", "P":"C1CC(NC1)C(=O)O", "Q":"C(CC(=O)N)C(C(=O)O)N", 
                   "R":"C(CC(C(=O)O)N)CN=C(N)N", "S":"C(C(C(=O)O)N)O", "T":"CC(C(C(=O)O)N)O", "V":"CC(C)C(C(=O)O)N", "Y":"C1=CC(=CC=C1CC(C(=O)O)N)O", "W":"C1=CC=C2C(=C1)C(=CN2)CC(C(=O)O)N"}
            try:
                return  amino_SMILE[aminoacid]
            except KeyError:
                return "SMILESnotfound" 

        X_tr = []
        X_tr.append(aacode)
        X_out = []
        for i in X_tr:
            temp = []
            for j in i:
                temp.extend(get_aaindex(j))
                SML1 = amino_to_SMILES (j)
                bitvectMorg = aminoMorgan_to30bits(SML1, int(FP_size_Amino))
                btvctNegPos = convert01_minus1plus1(bitvectMorg)
                temp.extend(btvctNegPos)
            X_out.append(temp)
        ret_NParray = np.array(X_out)
        return (ret_NParray)

    X = NRPS_training_data()

    code_inputNN = "preprocessed_NRPS_codes_inputforNNassc.p"  
    pickle.dump(X, open(code_inputNN, "wb"))

###################################################################################################

    def run_learned_model():
        from keras.models import Sequential 
        from keras.layers import Dense, Activation
        from keras.metrics import binary_accuracy
        import sys, codecs, pickle
        from keras.models import model_from_json
        #sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())

        # load json and create model
        json_file = open('./lib/Keras_NNassc_saved_model_final.py.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        
        # load weights into new model
        loaded_model.load_weights("./lib/Keras_NNassc_saved_model_final_weights.py.h5")

        # evaluate loaded model on test data
        loaded_model.compile(loss='binary_crossentropy', optimizer='RMSprop', metrics=['accuracy'])

        rare = open("preprocessed_NRPS_codes_inputforNNassc.p", "rb")
        rare_classes = pickle.load(rare, encoding='latin1')
        y_rare = loaded_model.predict(rare_classes) 
        prob_out = []
        for ij in y_rare:
            prob_out.append(ij.tolist())
        finalList = []

        ### Predicted probabilities (each bit position) to binary encoding - 0.5 cutoff ###
        ### if prob 0.0 - 0.5 or eauql to 0.5 - Bit is set to 1  otherwise 0 #### 
        bit_prob_cutoff = 0.5

        for iff in range(0, len(prob_out)):
            outlist = []
            temp = prob_out[iff]
            for jff in temp:
                roundedjff = round(jff, 3)
                if (float(roundedjff) >= bit_prob_cutoff): ##bit cutoff###
                    outlist.append(1)
                else:
                    outlist.append(0)

            finalList.append(outlist)
        pickle.dump(finalList, open("NNassc_output_SubstrateJaccardIndex.p", "wb"), protocol=2)
    run_learned_model()

######################################################################################################
	
    bitsize = int(1024)
    index = int(0)
    flname = "NNassc_output_SubstrateJaccardIndex.p"
    pickled_OUT = open(flname, "rb")
    NRPS_OUT = pickle.load(pickled_OUT) 

    ### substrates and SMILES strings pair - complete dataset ###
    train_test_dict = {'C(=CC(=O)O)C(=O)O':'fumaricacid',  'C(C(=O)O)N':'Glycine',  'C(C(C(=O)O)N)O':'Serine',  'C(C(C(=O)O)N)S':'Cysteine',  'C(CC(=O)N)C(C(=O)O)N':'Glutamine',  'C(CC(C(=O)O)N)CC(=O)O':'2-Aminoadipicacid',  'C(CC(C(=O)O)N)CN':'ornithine',  'C(CN)C(=O)O':'beta-Alanine',  'C(CO)C(C(=O)O)N':'Homoserine',  'C1=CC(=CC=C1CC(=O)C(=O)O)O':'4hydroxy-phenylpyruvicacid',  'C1=CC(=CC=C1CC(C(=O)O)N)O':'Tyrosine',  'C1=CC=C(C(=C1)C(=O)O)N':'anthranilicacid',  'C1=CC=C(C=C1)CC(=O)C(=O)O':'phenylpyruvicacid',  'C1=CC=C(C=C1)CC(C(=O)O)N':'Phenylalanine',  'C1=CC=C2C(=C1)C(=CN2)CC(=O)C(=O)O':'indole-3-pyruvicacid',  'C1=CC=C2C(=C1)C(=CN2)CC(C(=O)O)N':'Tryptophan',  'C1C(O1)C(=O)CCCCCC(C(=O)O)N':'2-amino-8-(oxiran-2-yl)-8-oxooctanoicacid',  'C1CC(NC1)C(=O)O':'Proline',  'C1CCNC(C1)C(=O)O':'homoproline',  'CC(=CC(=O)N(CCCC(C(=O)O)N)O)CCO':'ndeltacisanhydromevalonyl-ndeltahydroxy-ornithine',  'CC(C(=O)O)N':'Alanine',  'CC(C)C(C(=O)O)N':'Valine',  'CC(C)C(C(=O)O)O':'2-hydroxyisovalerate',  'CC(C)CC(C(=O)O)N':'Leucine',  'CC(C)CC(C(=O)O)O':'alphahydroxy-isocaprpoicacid',  'CC1=C(C=C(C(=C1C)C(=O)O)O)O':'5methyl-orselinicacid',  'CC1=CC(=C(C(=C1C(=O)O)O)CC=C(C)CCC=C(C)CCC=C(C)C)O':'grifolicacid',  'CCC(=O)CCCCCC(C(=O)O)N':'(2S)-2-amino-8-oxodecanoicacid',  'CCC(C(=O)O)N':'2-Aminobutyricacid',  'CCC(C)C(C(=O)O)N':'allo-Isoleucine',  'CCC(C)C(C(=O)O)O':'2-hydroxy-3-methylpentanoicacid',  'CON1C=C(C2=CC=CC=C21)CC(C(=O)O)N':'s-nmethoxy_trp', "C(CC(C(=O)O)N)CN=C(N)N":"Arginine", "C(CCN=C(N)N)CC(C(=O)O)N":"homoarginine",  "C1=CC=C(C=C1)C=CC(=O)O":"cinnamicacid", "C1=CC(=C(C=C1C=CC(=O)O)O)O":"caffeicacid", 'C1=CC=C(C(=C1)C(=O)O)N':'anthranilicacid', 'C1=CC=C(C=C1)CC(=O)C(=O)O':'phenylpyruvicacid', "C1C(CNC1C(=O)O)O":"Trans-4-Hydroxy-L-proline",  "C1CC(N=C1)C(=O)O":"1-pyrroline-5-carboxylicacid"}


    def tanimoto(v1, v2):
    ### Calculates tanimoto similarity for two bit vectors ### 
        return(np.bitwise_and(v1, v2).sum() / np.bitwise_or(v1, v2).sum())

    def SMILES2bitvect(SMILE, bitsize):
        ## Conversion of canonical SMILE strings into a bitVector ###
        from rdkit import Chem 
        from rdkit.Chem import AllChem
        mol = Chem.MolFromSmiles(SMILE)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=int(bitsize), useFeatures=True)
        fp_bitstr = fp.ToBitString()
        fp_bitstr_list = [int(P) for P in fp_bitstr]
        return (fp_bitstr_list)
    
    import numpy as np
    from numpy import dot
    from numpy.linalg import norm

    output_data = []
    out = NRPS_OUT
    out_N = np.array(out[index])
    desc_list = []

    ### Matching predicted bitvector with training dataset substrate bitvectors ###
    ## Tanimoto coefficient -  predicted and training dataset substrate bit vectors ##
    print("Input NRPS code residues: " + aacode)
    print("Output: ")
    print("Rank" + "\t" + "Tanimoto" + "\t" + "Substarte") 
    for sm1 in train_test_dict:
        tmp1 = np.array(SMILES2bitvect(sm1, bitsize))
        simlrt = round ( dot(tmp1, out_N)/(norm(tmp1)*norm(out_N)), 3)
        output_data.append((simlrt, train_test_dict[sm1]))
        ##Output : Rank(Tanimoto coefficient, Substarte)

    desc_list = sorted(output_data, reverse=True)

    xcount = 0
    ## Limit the number of substrate hits to 5 ###
    substrate_hits = 5
    for it in desc_list:
        xcount = xcount + 1
        if xcount <= substrate_hits:
            print (str(xcount) + ".\t" + str(it[0]) + "\t\t" + str(it[1]))
    desc_dict = { i : desc_list[i] for i in range(0, len(desc_list) ) }


def main(aacode):
    run_NNassc(aacode)

if __name__ == "__main__":
    main(aacode)

