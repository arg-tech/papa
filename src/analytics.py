import json
# from . import renewed_ova2_toolbox as ova2
# from . import renewed_ova3_toolbox as ova3
from . import xaif_toolbox as ova3
import sys
import os
from collections import Counter

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import csv
import re
nltk.data.path.append("tools/nltk_data")
import spacy
import en_core_web_sm
nlp = en_core_web_sm.load()
nlp = spacy.load("en_core_web_sm")

################
# MISC UTILITY #
################

# Get speaker who first introduced I-node based on the speaker of the chronologically 
# earliest locution connected to it
def i_node_introducer(i_node, all_nodes):
        intro_loc = all_nodes[i_node]['introby'][0]
        speaker = all_nodes[intro_loc]['speaker'][0] # !! ASSUMPTION
        # print("all_nodes[i_node]['introby'] = ", all_nodes[i_node]['introby'])
        # print("all_nodes[intro_loc]['speaker'] = ", all_nodes[intro_loc]['speaker'])
        return speaker


# Given the ID of an argument scheme node, return the ID of the L-node 
def arg_rel_lnode(rel_node, all_nodes):
    ya_node_in = [n for n in all_nodes if all_nodes[n]['type'] == 'YA' and rel_node in all_nodes[n]['eout']][0]
    ta_node_in = [n for n in all_nodes if all_nodes[n]['type'] == 'TA' and ya_node_in in all_nodes[n]['eout']][0]

    l_nodes_out = [n for n in all_nodes if all_nodes[n]['type'] == 'L' and ta_node_in in all_nodes[n]['ein']]
    if len(l_nodes_out) == 1:
        return l_nodes_out[0]
    else:
        # if multiple outgoing, find the one with an I-node connected to the relation
        for l in l_nodes_out:
            ya_node_out = [n for n in all_nodes if all_nodes[n]['type'] == 'YA' and l in all_nodes[n]['eout']]
            for ya in ya_node_out:
                connected_i_nodes = [n for n in all_nodes if all_nodes[n]['type'] == 'I' and ya in all_nodes[n]['ein'] 
                                    and (rel_node in all_nodes[n]['ein'] or rel_node in all_nodes[n]['eout'])]
                # connected_i = [n for n in all_nodes if l in all_nodes[n]['']]
                if len(connected_i_nodes) != 0:
                    return l

# Given the ID of an I-node, return the IDs of any I-nodes it rephrases
def rephrased_by(i_node, all_nodes, debug=False):
    rephrased_by_i = []
    ma_nodes = [n for n in all_nodes if all_nodes[n]['type'] == 'MA' and i_node in all_nodes[n]['ein']]
    if debug:
        if len(ma_nodes) == 0:
            print(f"\t\tNo MA nodes connected to I-node {i_node}")
        else:
            print(f"\t\tMA nodes connected to I-node {i_node}:", ma_nodes)

    for ma in ma_nodes:
        # prop being rephrased
        rephrased_by_i += [n for n in all_nodes if all_nodes[n]['type'] == 'I' and n in all_nodes[ma]['eout']]
    return rephrased_by_i

def props_linked_to_rel(rel_id, all_nodes):
    i_nodes = [n for n in all_nodes 
               if (n in all_nodes[rel_id]['eout'] or n in all_nodes[rel_id]['ein'])
                and all_nodes[n]['type'] == 'I']
    return i_nodes

# return T/F for whether the node queried has an outgoing edge to a node with the provided text
def node_anchors_text(node_id, node_text, all_nodes):
    anchored = [n for n in all_nodes[node_id]['eout'] if all_nodes[n]['text'] == node_text]
    return len(anchored) > 0

# Return ID of I-node (if any) anchored in L-node, possibly via a reported speech chain
def i_from_l_node(l_node_id, all_nodes):
    # Go through nodes from the L-node until YA found
    for e_out in all_nodes[l_node_id]['eout']:
        if all_nodes[e_out]['type'] == 'YA':
            ya = e_out
            # Check for L or I from this YA: if I then return; if L then continue down chain
            for ya_out in all_nodes[ya]['eout']:
                if all_nodes[ya_out]['type'] == 'I':
                    return ya_out
                elif all_nodes[ya_out]['type'] == 'L':
                    return i_from_l_node(ya_out, all_nodes)
    # If none found, return empty string
    return ''


########
# MONO #
########


########################
# Counts and densities #
######################## 

# Unrefined return of text field wordcount
def map_wordcount(xaif):
        text = xaif['text']
        # Remove char codes and tags
        text = re.sub("<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});", '', text)

        text = text.replace('\n', ' ')
        return {'wordcount': len(text.split())}


# Wordcounts from L-nodes only, excluding whatever comes before first colon
# (assumed to be speaker name)
def loc_wordcount(xaif, speaker=False, verbose=False):
    all_nodes, said = ova3.xaif_preanalytic_info_collection(xaif)
    l_nodes = [n for n in all_nodes if all_nodes[n]['type'] == 'L']
    
    loc_wordcounts = {}
    if speaker:
        for spkr in said.keys():
            loc_wordcounts[spkr] = {}
            spkr_l_nodes = [n for n in l_nodes if spkr in all_nodes[n]['speaker']]
            wordcount = 0
            for l in spkr_l_nodes:
                splits = all_nodes[l]['text'].split(':')
                if len(splits) > 2:
                    print(f"L-node {l} cannot be split on ':'")
                text = ' '.join(splits[1:])
                if verbose:
                    print(f"{len(text.split())}: {all_nodes[l]['text']}")

                wordcount += len(text.split())
            loc_wordcounts[spkr]['loc_wordcount'] = wordcount
    
    else:
        wordcount = 0
        for l in l_nodes:
            splits = all_nodes[l]['text'].split(':')
            if len(splits) > 2:
                print(f"L-node {l} cannot be split on ':'")
            text = ' '.join(splits[1:])
            wordcount += len(text.split())
            if verbose:
                print(f"{len(text.split())}: {all_nodes[l]['text']}")
        loc_wordcounts['loc_wordcount'] = wordcount
    
    return loc_wordcounts



def loc_counts(xaif, speaker=False, verbose=False):
    if 'AIF' in xaif.keys():
        all_nodes, said = ova3.xaif_preanalytic_info_collection(xaif)
    else:
        all_nodes, _ = ova2.xaif_preanalytic_info_collection(xaif)
    
    # relation_counts = arg_relation_counts(xaif)
    
    if speaker:
        spkr_loc_counts = {}
        for s in said.keys():
            spkr_locs = len([n for n in all_nodes if all_nodes[n]['type'] == 'L' and all_nodes[n]['speaker'][0] == s])
            spkr_loc_counts[s] = {}
            spkr_loc_counts[s]['loc_count'] = spkr_locs
        return spkr_loc_counts
    else: # Avoiding any reported speech: assumes meta-nodes for analysis have been removed
        # get all YA and L nodes
        ya_nodes = [n for n in all_nodes if all_nodes[n]['type'] == 'YA']
        l_nodes = [n for n in all_nodes if all_nodes[n]['type'] == 'L']
        
        if verbose:
            print(f"{len(ya_nodes)} YA-nodes:", ya_nodes)
            print(f"{len(l_nodes)} L-nodes:", l_nodes)
            for l in l_nodes:
                print(f"L-node {l} incoming:")
                for n in all_nodes[l]['ein']:
                    print(f"\t{n}")
                    if not (set(all_nodes[n]['ein']).intersection(set(ya_nodes))):
                        print('\tNo intersection with YAs')

        # Cut any l-nodes with an incoming YA: will be reported speech
        # i.e. only keep l-nodes *without* a YA node among their incoming nodes
        l_orig_nodes = [n for n in l_nodes if not set(all_nodes[n]['ein']).intersection(set(ya_nodes))]

        loc_counts = {'loc_count': len(l_orig_nodes)}
        return loc_counts



def arg_relation_counts(xaif, speaker=False, verbose=False, skip_altgive=False):
    relation_counts = {}
    if 'AIF' in xaif.keys():
        all_nodes, said = ova3.xaif_preanalytic_info_collection(xaif)
    else:
        all_nodes, said = ova2.xaif_preanalytic_info_collection(xaif)

    ra_nodes = [n for n in all_nodes if all_nodes[n]['type'] == 'RA']
    ca_nodes = [n for n in all_nodes if all_nodes[n]['type'] == 'CA']
    ma_nodes = [n for n in all_nodes if all_nodes[n]['type'] == 'MA']

    # remove any CAs anchored by Alternative Giving
    if skip_altgive:
        alt_gives = [n for n in all_nodes 
                        if all_nodes[n]['type'] == 'YA' and all_nodes[n]['text'] == 'Alternative Giving']
        if verbose:
            print(f"Alt-giving nodes: ", alt_gives)
        # Remove any CAs with an incoming edge from an Alternative Giving node
        ca_nodes = [n for n in ca_nodes if not set(all_nodes[n]['ein']).intersection(alt_gives)]

    if speaker:
        if verbose:
            print(f"{len(ra_nodes)} RAs found")
            print(f"{len(ca_nodes)} CAs found")
            print(f"{len(ma_nodes)} MAs found")
            print("Speakers in 'said': ", list(said.keys()))
        for spkr in said:
            spkr_ra_all = [n for n in ra_nodes if spkr in all_nodes[n]['speaker']]
            spkr_ca_all = [n for n in ca_nodes if spkr in all_nodes[n]['speaker']]
            spkr_ma_all = [n for n in ma_nodes if spkr in all_nodes[n]['speaker']]
            
            relation_counts[spkr] = {}
            relation_counts[spkr]['ra_count'] = len(spkr_ra_all)
            relation_counts[spkr]['ca_count'] = len(spkr_ca_all)
            relation_counts[spkr]['ma_count'] = len(spkr_ma_all)

    else:
        relation_counts['ra_count'] = len(ra_nodes)
        relation_counts['ca_count'] = len(ca_nodes)
        relation_counts['ma_count'] = len(ma_nodes)

    return relation_counts

def ra_ca_ratio(xaif, speaker=False, verbose=False, skip_altgive=False):
    rel_counts = arg_relation_counts(xaif, speaker=speaker, skip_altgive=skip_altgive)
    
    ra_to_ca = {}
    
    if not speaker:
        if rel_counts['ca_count'] == 0:
            ra_to_ca['ra_to_ca'] = 0
        else:
            ra_to_ca['ra_to_ca'] = rel_counts['ra_count']/rel_counts['ca_count']

    else:
        for spkr in rel_counts:
            ra_to_ca[spkr] = {}
            if rel_counts[spkr]['ca_count'] == 0:
                ra_to_ca[spkr]['ra_to_ca'] = 0
            else:
                ra_to_ca[spkr]['ra_to_ca'] = rel_counts[spkr]['ra_count']/rel_counts[spkr]['ca_count']

    return ra_to_ca

# Density based on textfield wordcount
def arg_word_densities(xaif, speaker=False, verbose=False, skip_altgive=False):
    relation_counts = arg_relation_counts(xaif, speaker=speaker, skip_altgive=skip_altgive)

    if speaker:
        if 'AIF' in xaif.keys():
            spkr_wordcounts = ova3.spkr_wordcounts(xaif)
        else:
            spkr_wordcounts = ova2.spkr_wordcounts(xaif)

        if verbose:
            print("Keys in relation counts: ", relation_counts.keys())
            print("Wordcounts: ", spkr_wordcounts)
            print(relation_counts)
        for s in spkr_wordcounts.keys():
            if spkr_wordcounts[s]['wordcount'] != 0 and s in relation_counts:
                relation_counts[s]['ra_word_density'] = relation_counts[s]['ra_count']/spkr_wordcounts[s]['wordcount']
                relation_counts[s]['ca_word_density'] = relation_counts[s]['ca_count']/spkr_wordcounts[s]['wordcount']
                relation_counts[s]['ma_word_density'] = relation_counts[s]['ma_count']/spkr_wordcounts[s]['wordcount']
                relation_counts[s]['arg_word_density'] = (relation_counts[s]['ra_count'] + relation_counts[s]['ca_count'])/spkr_wordcounts[s]['wordcount']

            else:
                if s not in relation_counts:
                    relation_counts[s] = {}

                    relation_counts[s]['ra_count'] = 0
                    relation_counts[s]['ca_count'] = 0
                    relation_counts[s]['ma_count'] = 0

                relation_counts[s]['ra_word_density'] = 0
                relation_counts[s]['ca_word_density'] = 0
                relation_counts[s]['ma_word_density'] = 0
                relation_counts[s]['arg_word_density'] = 0
        
    else:
        wordcount = map_wordcount(xaif)['wordcount']
        if wordcount != 0:
            relation_counts['ra_word_density'] = relation_counts['ra_count']/wordcount
            relation_counts['ca_word_density'] = relation_counts['ca_count']/wordcount
            relation_counts['ma_word_density'] = relation_counts['ma_count']/wordcount
            relation_counts['arg_word_density'] = (relation_counts['ra_count'] + relation_counts['ca_count'])/wordcount
        else:
            relation_counts['ra_word_density'] = 0
            relation_counts['ca_word_density'] = 0
            relation_counts['ma_word_density'] = 0
            relation_counts['arg_word_density'] = 0

    return relation_counts

# Density based on number of locutions
def arg_loc_densities(xaif, speaker=False, verbose=False, skip_altgive=False):
    relation_counts = arg_relation_counts(xaif, speaker=speaker, skip_altgive=skip_altgive)
    l_counts = loc_counts(xaif, speaker=speaker)

    # if 'AIF' in xaif.keys():
    #     all_nodes, _ = ova3.xaif_preanalytic_info_collection(xaif)
    # else:
    #     all_nodes, _ = ova2.xaif_preanalytic_info_collection(xaif)
    
    if speaker:
        if verbose:
            print("Speakers listed in relation_counts:", list(relation_counts.keys()))

        for s in relation_counts.keys():
            # spkr_locs = len([n for n in all_nodes if all_nodes[n]['type'] == 'L' and all_nodes[n]['speaker'][0] == s])
            spkr_locs = l_counts[s]['loc_count']

            if spkr_locs != 0:
                relation_counts[s]['ra_loc_density'] = relation_counts[s]['ra_count']/spkr_locs
                relation_counts[s]['ca_loc_density'] = relation_counts[s]['ca_count']/spkr_locs
                relation_counts[s]['ma_loc_density'] = relation_counts[s]['ma_count']/spkr_locs
                relation_counts[s]['arg_loc_density'] = (relation_counts[s]['ra_count'] + relation_counts[s]['ca_count'])/spkr_locs
            else:
                relation_counts[s]['ra_loc_density'] = 0
                relation_counts[s]['ca_loc_density'] = 0
                relation_counts[s]['ma_loc_density'] = 0
                relation_counts[s]['arg_loc_density'] = 0
    else:
        if l_counts['loc_count'] != 0:
            relation_counts['ra_loc_density'] = relation_counts['ra_count']/l_counts['loc_count']
            relation_counts['ca_loc_density'] = relation_counts['ca_count']/l_counts['loc_count']
            relation_counts['ma_loc_density'] = relation_counts['ma_count']/l_counts['loc_count']
            relation_counts['arg_loc_density'] = (relation_counts['ra_count'] + relation_counts['ca_count'])/l_counts['loc_count']
        else:
            relation_counts['ra_loc_density'] = 0
            relation_counts['ca_loc_density'] = 0
            relation_counts['ma_loc_density'] = 0
            relation_counts['arg_loc_density'] = 0

    return relation_counts


# Density relative to wordcount from locutions
def arg_locword_densities(xaif, speaker=False, verbose=False, skip_altgive=False):
    relation_counts = arg_relation_counts(xaif, speaker=speaker, skip_altgive=skip_altgive)

    if speaker:
        spkr_wordcounts = loc_wordcount(xaif, speaker=True)

        if verbose:
            print("Keys in relation counts: ", relation_counts.keys())
            print("L-node wordcounts: ", spkr_wordcounts)
            print(relation_counts)
        for s in spkr_wordcounts.keys():
            if spkr_wordcounts[s]['loc_wordcount'] != 0 and s in relation_counts:
                relation_counts[s]['ra_locword_density'] = relation_counts[s]['ra_count']/spkr_wordcounts[s]['loc_wordcount']
                relation_counts[s]['ca_locword_density'] = relation_counts[s]['ca_count']/spkr_wordcounts[s]['loc_wordcount']
                relation_counts[s]['ma_locword_density'] = relation_counts[s]['ma_count']/spkr_wordcounts[s]['loc_wordcount']
                relation_counts[s]['arg_locword_density'] = (relation_counts[s]['ra_count'] + relation_counts[s]['ca_count'])/spkr_wordcounts[s]['loc_wordcount']

            else:
                if s not in relation_counts:
                    relation_counts[s] = {}

                    relation_counts[s]['ra_count'] = 0
                    relation_counts[s]['ca_count'] = 0
                    relation_counts[s]['ma_count'] = 0

                relation_counts[s]['ra_locword_density'] = 0
                relation_counts[s]['ca_locword_density'] = 0
                relation_counts[s]['ma_locword_density'] = 0
                relation_counts[s]['arg_locword_density'] = 0
        
    else:
        wordcount = loc_wordcount(xaif, speaker=False)['loc_wordcount']
        if wordcount != 0:
            relation_counts['ra_locword_density'] = relation_counts['ra_count']/wordcount
            relation_counts['ca_locword_density'] = relation_counts['ca_count']/wordcount
            relation_counts['ma_locword_density'] = relation_counts['ma_count']/wordcount
            relation_counts['arg_locword_density'] = (relation_counts['ra_count'] + relation_counts['ca_count'])/wordcount
        else:
            relation_counts['ra_locword_density'] = 0
            relation_counts['ca_locword_density'] = 0
            relation_counts['ma_locword_density'] = 0
            relation_counts['arg_locword_density'] = 0

    return relation_counts


def restating_count(xaif, speaker=False, verbose=False):
    restating_counts = {}
    
    if not speaker:
        restatement = [n for n in xaif['AIF']['nodes'] if n['type'] == 'YA' and n['text'] == 'Restating']
        restating_counts['restating_count'] = len(restatement)
    else:
        all_nodes, said = ova3.xaif_preanalytic_info_collection(xaif)
        all_restatement = [n for n in all_nodes if all_nodes[n]['type'] == 'YA' and all_nodes[n]['text'] == 'Restating']

        for spkr in said:
            spkr_restates = [n for n in all_restatement if spkr in all_nodes[n]['speaker']]
            restating_counts[spkr] = {'restating_count': len(spkr_restates)}
    
    return restating_counts


# I-node is a premise if it has an outgoing edge to an RA
# When speaker-wise: only count if premise to a conclusion by same speaker, or to a previously given other-speaker conclusion
# (i.e. don't attribute premise-making to a speaker unless it's part of their own argument construction)
def premise_count(xaif, speaker=False, verbose=False):
    all_nodes, said = ova3.xaif_preanalytic_info_collection(xaif)

    ra_nodes = [n for n in all_nodes if all_nodes[n]['type'] == 'RA']

    premise_count = {}

    if speaker:
        for spkr in said:
            premise_count[spkr] = {'premise_count': 0}

            # speaker i-nodes
            i_nodes = [n for n in all_nodes if all_nodes[n]['type'] == 'I' and spkr in all_nodes[n]['saidby']]
            
            # for each, check if it's a premise
            # if a premise, check if for something that is same-speaker or chron earlier -> if yes to either, is a premise
            for i in i_nodes:
                is_premise=False
                # Get relations in which this I is a premise
                premise_for = [ra for ra in all_nodes[i]['eout'] if all_nodes[ra]['type'] == 'RA']
                if len(premise_for) == 0:
                    continue

                # Check if any RA from the I-node concludes in an I-node that meets the criteria above
                for ra in premise_for:
                    concl_nodes = [j for j in all_nodes[ra]['eout'] if all_nodes[j]['type'] == 'I']
                    
                    for n in concl_nodes:
                        # Same speaker, speaker's own argument, is premise whether initial or not
                        if spkr == i_node_introducer(n, all_nodes):
                            is_premise = True
                            break
                        # Different speaker: check if concl is pre-existing material -- if concl added later, don't consider this a spkr premise
                        else:
                            intro_prem = all_nodes[i]['introby'][0]
                            intro_concl = all_nodes[n]['introby'][0]
                            if all_nodes[intro_concl]['chron'] < all_nodes[intro_prem]['chron']:
                                is_premise = True

                    if is_premise:
                        break
                
                if is_premise:
                    premise_count[spkr]['premise_count'] += 1

    
    else:
        i_nodes = [n for n in all_nodes if all_nodes[n]['type'] == 'I']
        
        # I-node is a premise if there is an intersection between its outgoing edge targets and the list of RAs
        premise_i_nodes = [n for n in i_nodes if set(all_nodes[n]['eout']) & set(ra_nodes)]
        premise_count['premise_count'] = len(premise_i_nodes)

    return premise_count
    

def concl_count(xaif, speaker=False, verbose=False):
    all_nodes, said = ova3.xaif_preanalytic_info_collection(xaif)
    ra_nodes = [n for n in all_nodes if all_nodes[n]['type'] == 'RA']

    conclusion_count = {}

    if speaker:
        for spkr in said:
            conclusion_count[spkr] = {'concl_count': 0}

            # speaker i-nodes
            i_nodes = [n for n in all_nodes if all_nodes[n]['type'] == 'I' and spkr in all_nodes[n]['saidby']]
            
            # for each, check if it's a conclusion
            # if a conclusion, check if for something that is same-speaker or has earlier chron -> if yes to either, is a conclusion
            for i in i_nodes:
                is_concl=False
                # Get relations in which this I is a conclusion
                concl_for = [ra for ra in all_nodes[i]['ein'] if all_nodes[ra]['type'] == 'RA']
                if len(concl_for) == 0:
                    continue

                # Check if any RA from the I-node derives from an I-node that meets the criteria above
                for ra in concl_for:
                    premise_nodes = [j for j in all_nodes[ra]['ein'] if all_nodes[j]['type'] == 'I']
                    
                    for n in premise_nodes:
                        # Same speaker, speaker's own argument, is premise whether initial or not
                        if spkr == i_node_introducer(n, all_nodes):
                            is_concl = True
                            break
                        # Different speaker: check if premise is pre-existing material -- if premise added later, don't consider this a spkr concl
                        else:
                            intro_concl = all_nodes[i]['introby'][0]
                            intro_prem = all_nodes[n]['introby'][0]
                            if all_nodes[intro_prem]['chron'] < all_nodes[intro_concl]['chron']:
                                is_concl = True

                    if is_concl:
                        break
                
                if is_concl:
                    conclusion_count[spkr]['concl_count'] += 1

    
    else:
        i_nodes = [n for n in all_nodes if all_nodes[n]['type'] == 'I']
        
        # I-node is a conclusion if there is an intersection between its incoming edge origins and the list of RAs
        conclusion_i_nodes = [n for n in i_nodes if set(all_nodes[n]['ein']) & set(ra_nodes)]
        conclusion_count['concl_count'] = len(conclusion_i_nodes)

    return conclusion_count


def prem_concl_ratio(xaif, speaker=False, verbose=False):
    prems = premise_count(xaif, speaker=speaker)
    concs = concl_count(xaif, speaker=speaker)

    prem_to_conc = {}

    if not speaker:
        if concs['concl_count'] == 0:
            prem_to_conc['premise_to_concl'] = 0
        else:
            prem_to_conc['premise_to_concl'] = prems['premise_count']/concs['concl_count']
    
    else:
        for spkr in prems:
            prem_to_conc[spkr] = {}
            if concs[spkr]['concl_count'] == 0:
                prem_to_conc[spkr]['premise_to_concl'] = 0
            else:
                prem_to_conc[spkr]['premise_to_concl'] = prems[spkr]['premise_count']/concs[spkr]['concl_count']

    return prem_to_conc




#######################
# Order and structure # 
#######################

def concl_first_perc(xaif, speaker=False):
    if 'AIF' in xaif.keys():
        all_nodes, said = ova3.xaif_preanalytic_info_collection(xaif)
    else:
        all_nodes, said = ova2.xaif_preanalytic_info_collection(xaif)
    ra_nodes = [n for n in all_nodes if all_nodes[n]['type'] == 'RA']

    # Per speaker
    concl_first = {}
    
    if speaker:
        for spkr in said:
            spkr_ra_all = [n for n in ra_nodes if spkr in all_nodes[n]['speaker']]
            ra_total = len(spkr_ra_all)
            
            concl_first[spkr] = {}
            concl_first[spkr]['ra_count'] = ra_total
            concl_first[spkr]['ra_concl_first'] = 0
            concl_first[spkr]['ra_concl_first_perc'] = 0.0
            
            if ra_total == 0:
                continue

            for ra in spkr_ra_all:
                found_order = False
                for ra_incoming_node in all_nodes[ra]['ein']:

                    # Found a YA with edge to the RA
                    if all_nodes[ra_incoming_node]['type'] == 'YA':
                        
                        # all_nodes[ra_incoming_node] is currently a YA:
                        # Get nodes with edge leading into the YA: s
                        # Get the TA with edge to the YA: YA should only have TA incoming if it anchors RA
                        for ya_incoming_node in all_nodes[ra_incoming_node]['ein']: 
                            # YA incoming node should be TA
                            # Get L-node descended from TA
                            for ta_out in all_nodes[ya_incoming_node]['eout']:
                                if all_nodes[ta_out]['type'] == 'L':
                                    l = ta_out
                                    i = i_from_l_node(l, all_nodes)
                                    if ra in all_nodes[i]['eout']:
                                        concl_first[spkr]['ra_concl_first'] += 1
                                        found_order = True
                                    elif ra in all_nodes[i]['ein']:
                                        found_order = True
                                    else:
                                        print('I-node not connected to RA')
                                        print(f"\t\t{i}: {all_nodes[i]['text']}")
                                        # print(f"Disconnected I-node ")
            
            concl_first[spkr]['ra_concl_first_perc'] = concl_first[spkr]['ra_concl_first']/concl_first[spkr]['ra_count']
    
    # This should be refactored
    else:
        ra_total = len(ra_nodes)
        
        concl_first['ra_count'] = ra_total
        concl_first['ra_concl_first'] = 0
        concl_first['ra_concl_first_perc'] = 0.0
        
        if ra_total == 0:
            return concl_first
        
        for ra in ra_nodes:
            for ra_incoming_node in all_nodes[ra]['ein']:

                # Found a YA with edge to the RA
                if all_nodes[ra_incoming_node]['type'] == 'YA':
                    
                    # all_nodes[ra_incoming_node] is currently a YA:
                    # Get nodes with edge leading into the YA: s
                    # Get the TA with edge to the YA: YA should only have TA incoming if it anchors RA
                    for ya_incoming_node in all_nodes[ra_incoming_node]['ein']: 
                        # YA incoming node should be TA
                        # Get L-node descended from TA
                        for ta_out in all_nodes[ya_incoming_node]['eout']:
                            if all_nodes[ta_out]['type'] == 'L':
                                l = ta_out
                                i = i_from_l_node(l, all_nodes)
                                if ra in all_nodes[i]['eout']:
                                    concl_first['ra_concl_first'] += 1
                                else:
                                    print('I-node not connected to RA')
                                    print(f"\t\t{i}: {all_nodes[i]['text']}")
        
        concl_first['ra_concl_first_perc'] = concl_first['ra_concl_first']/concl_first['ra_count']

    return concl_first


def ra_in_serial(xaif, speaker=False, verbose=False):
    if 'AIF' in xaif.keys():
        all_nodes, said = ova3.xaif_preanalytic_info_collection(xaif)
    else:
        all_nodes, said = ova2.xaif_preanalytic_info_collection(xaif)
    ra_nodes = [n for n in all_nodes if all_nodes[n]['type'] == 'RA']

    # Per speaker
    ra_serial = {}
    
    if speaker:
        for spkr in said:
            if verbose:
                print("Checking speaker ", spkr)
            ra_serial[spkr] = {}
            spkr_ra_all = [n for n in ra_nodes if spkr in all_nodes[n]['speaker']]

            if verbose:
                print("\tSpeaker's RA nodes: ", spkr_ra_all)

            ra_serial[spkr]['ra_serial_count'] = 0

            if len(spkr_ra_all) == 0:
                ra_serial[spkr]['ra_serial_perc'] = 0.0
                continue

            for ra in spkr_ra_all:
                if verbose:
                    print("\tChecking RA ", ra)
                serial = False
                # Get I-nodes with edge outgoing toward the RA
                i_nodes_to_ra = [n for n in all_nodes if ra in all_nodes[n]['eout'] and all_nodes[n]['type'] == 'I']

                if verbose:
                    print(f"\t\tI-nodes premise of (with edge toward) RA {ra}:")

                # Check each of these for an incoming RA by the same speaker
                for i in i_nodes_to_ra:                
                    ra_nodes_to_i_node = [n for n in all_nodes if i in all_nodes[n]['eout'] and all_nodes[n]['type'] == 'RA']
                    if verbose:
                        print(f"\t\t\t{i}: {all_nodes[i]['text']}")
                        print(f"\t\t\tRA nodes where {i} is conclusion: ", ra_nodes_to_i_node)
                    
                    spkr_ra_match = [n for n in ra_nodes_to_i_node if spkr in all_nodes[n]['speaker']]
                    if verbose:
                        print(f"\t\t\tSubset with speaker match: ", spkr_ra_match)
                        print(f"\t\t\tVerified:")
                        for n in ra_nodes_to_i_node:
                            print(f"\t\t\t\t", all_nodes[n])
                    
                    # Serial found for this RA!
                    if len(spkr_ra_match) != 0:
                        serial = True
                        ra_serial[spkr]['ra_serial_count'] += 1
                        if verbose:
                            print(f'\t\t\t\tFound RA node(s) by same speaker with conclusion I-node {i}')
                            print(f'\t\t\t\t\t', spkr_ra_match)
                        break
                
                # If serial not found yet, check other direction
                if not serial:
                    # Get I-nodes with edge incoming from the RA
                    i_nodes_from_ra = [n for n in all_nodes if ra in all_nodes[n]['ein'] and all_nodes[n]['type'] == 'I']
                    
                    if verbose:
                        print(f"\t\tI-nodes conclusion of (with edge from) RA {ra}:")
                        for i in i_nodes_from_ra:
                            print(f"\t\t\t{i}: {all_nodes[i]['text']}")
                        print(f"\t\t\t{i}: {all_nodes[i]['text']}")
                        print(f"\t\t\tRA nodes where {i} is conclusion: ", ra_nodes_to_i_node)

                    # Check each of these for an incoming RA by the same 
                    for i in i_nodes_from_ra:
                        ra_nodes_from_i_node = [n for n in all_nodes if i in all_nodes[n]['ein'] and all_nodes[n]['type'] == 'RA']
                        spkr_ra_match = [n for n in ra_nodes_from_i_node if spkr in all_nodes[n]['speaker']]
                        
                        # I-node which is conclusion of RA has another RA from same speaker where it is premise
                        if len(spkr_ra_match) != 0:
                            ra_serial[spkr]['ra_serial_count'] += 1
                            if verbose:
                                print(f'\t\t\t\tFound RA node(s) by same speaker with premise I-node {i}')
                                print(f'\t\t\t\t\t', spkr_ra_match)
                        break

            ra_serial[spkr]['ra_serial_perc'] = ra_serial[spkr]['ra_serial_count']/len(spkr_ra_all)
    
    else:
        ra_serial['ra_serial_count'] = 0

        if len(ra_nodes) == 0:
            ra_serial['ra_serial_perc'] = 0.0
            return ra_serial

        for ra in ra_nodes:
            if verbose:
                print("\tChecking RA ", ra)
            serial = False
            
            # Get I-nodes with edge outgoing toward the RA
            i_nodes_to_ra = [n for n in all_nodes if ra in all_nodes[n]['eout'] and all_nodes[n]['type'] == 'I']

            if verbose:
                print(f"\t\tI-nodes premise of (with edge toward) RA {ra}:")

            # Check each of these for an incoming RA
            for i in i_nodes_to_ra:                
                ra_nodes_to_i_node = [n for n in all_nodes if i in all_nodes[n]['eout'] and all_nodes[n]['type'] == 'RA']
                if verbose:
                    print(f"\t\t\t{i}: {all_nodes[i]['text']}")
                    print(f"\t\t\tRA nodes where {i} is conclusion: ", ra_nodes_to_i_node)
                
                # If the I-node also has an incoming RA, orig RA is part of a serial arg!
                if len(ra_nodes_to_i_node) != 0:
                    serial = True
                    ra_serial['ra_serial_count'] += 1
                    if verbose:
                        print(f'\t\t\t\tFound RA node(s) with conclusion I-node {i}')
                        print(f'\t\t\t\t\t', ra_nodes_to_i_node)
                    break
            
            # If serial not found yet, check other direction
            if not serial:
                # Get I-nodes with edge incoming from the RA
                i_nodes_from_ra = [n for n in all_nodes if ra in all_nodes[n]['ein'] and all_nodes[n]['type'] == 'I']
                
                if verbose:
                    print(f"\t\tI-nodes conclusion of (with edge from) RA {ra}:")
                    for i in i_nodes_from_ra:
                        print(f"\t\t\t{i}: {all_nodes[i]['text']}")
                    print(f"\t\t\t{i}: {all_nodes[i]['text']}")
                    print(f"\t\t\tRA nodes where {i} is conclusion: ", ra_nodes_to_i_node)

                # Check each of these for an incoming RA by the same 
                for i in i_nodes_from_ra:
                    ra_nodes_from_i_node = [n for n in all_nodes if i in all_nodes[n]['ein'] and all_nodes[n]['type'] == 'RA']
                    
                    # I-node which is conclusion of RA has another RA for which it is premise
                    if len(ra_nodes_from_i_node) != 0:
                        ra_serial['ra_serial_count'] += 1
                        if verbose:
                            print(f'\t\t\t\tFound RA node(s) with premise I-node {i}')
                            print(f'\t\t\t\t\t', ra_nodes_from_i_node)
                    break

        ra_serial['ra_serial_perc'] = ra_serial['ra_serial_count']/len(ra_nodes)

    return ra_serial


def ra_in_convergent(xaif, speaker=False, verbose=False):
    if 'AIF' in xaif.keys():
        all_nodes, said = ova3.xaif_preanalytic_info_collection(xaif)
    else:
        all_nodes, said = ova2.xaif_preanalytic_info_collection(xaif)
    ra_nodes = [n for n in all_nodes if all_nodes[n]['type'] == 'RA']

    if verbose:
        print("All RA nodes in map: ", ra_nodes)
        for n in ra_nodes:
            print(f"\t{n}: {all_nodes[n]}")

    # Per speaker
    ra_convergent = {}

    if speaker:
        for spkr in said:
            ra_convergent[spkr] = {}
            spkr_ra_all = [n for n in ra_nodes if spkr in all_nodes[n]['speaker']]

            if verbose:
                print("Checking for ", spkr)
                print(f"\tRAs by {spkr}: ", spkr_ra_all)

            ra_convergent[spkr]['ra_convergent_count'] = 0

            if len(spkr_ra_all) == 0:
                ra_convergent[spkr]['ra_convergent_perc'] = 0.0
                continue

            for ra in spkr_ra_all:
                # get conclusion I-node(s) of the relation
                i_nodes_from_ra = [n for n in all_nodes if ra in all_nodes[n]['ein'] and all_nodes[n]['type'] == 'I']

                for i in i_nodes_from_ra:
                    # Get all incoming RAs to the conclusion which are by current speaker
                    ra_nodes_to_i = [n for n in all_nodes if i in all_nodes[n]['eout'] and all_nodes[n]['type'] == 'RA' and spkr in all_nodes[n]['speaker']]

                    # If there's more than one RA with this conclusion, then this RA is part of a convergent arg
                    # all the info we need about this RA, can move to next one
                    if len(ra_nodes_to_i) > 1:
                        ra_convergent[spkr]['ra_convergent_count'] += 1
                        break
            if verbose:
                print(f"ra_convergent[spkr]['ra_convergent_perc'] = {ra_convergent[spkr]['ra_convergent_count']}/{len(spkr_ra_all)}")
            ra_convergent[spkr]['ra_convergent_perc'] = ra_convergent[spkr]['ra_convergent_count']/len(spkr_ra_all)
    
    else:
        ra_convergent['ra_convergent_count'] = 0
        ra_convergent['ra_convergent_perc'] = 0.0
        if len(ra_nodes) == 0:
            return ra_convergent

        for ra in ra_nodes:
            i_nodes_from_ra = [n for n in all_nodes if ra in all_nodes[n]['ein'] and all_nodes[n]['type'] == 'I']
            for i in i_nodes_from_ra:
                # Get all incoming RAs to the conclusion
                ra_nodes_to_i = [n for n in all_nodes if i in all_nodes[n]['eout'] and all_nodes[n]['type'] == 'RA']
                
                # Multiple RAs with same conclusion as this RA, thus current RA is part of a convergent argument, add to count and move on
                if len(ra_nodes_to_i) > 1:
                    ra_convergent['ra_convergent_count'] += 1
                    break
        
        ra_convergent['ra_convergent_perc'] = ra_convergent['ra_convergent_count']/len(ra_nodes)


    return ra_convergent


def ra_in_divergent(xaif, speaker=False, verbose=False):
    if 'AIF' in xaif.keys():
        all_nodes, said = ova3.xaif_preanalytic_info_collection(xaif)
    else:
        all_nodes, said = ova2.xaif_preanalytic_info_collection(xaif)
    ra_nodes = [n for n in all_nodes if all_nodes[n]['type'] == 'RA']

    # Per speaker
    ra_divergent = {}

    if speaker:
        for spkr in said:
            ra_divergent[spkr] = {}
            spkr_ra_all = [n for n in ra_nodes if spkr in all_nodes[n]['speaker']]

            if verbose:
                print("Checking for ", spkr)
                print(f"\tRAs by {spkr}: ", spkr_ra_all)

            ra_divergent[spkr]['ra_divergent_count'] = 0

            if len(spkr_ra_all) == 0:
                ra_divergent[spkr]['ra_divergent_perc'] = 0.0
                continue

            for ra in spkr_ra_all:
                # get premise I-node of the relation
                i_node_to_ra = [n for n in all_nodes if ra in all_nodes[n]['eout'] and all_nodes[n]['type'] == 'I']

                if verbose:
                    print(f"\t\tChecking RA {ra}")
                    print(f"\t\t\tI-nodes to RA: ", i_node_to_ra)

                for i in i_node_to_ra:
                    # Get all outgoing RAs of the premise which are by current speaker
                    ra_nodes_from_i = [n for n in all_nodes if i in all_nodes[n]['ein'] and all_nodes[n]['type'] == 'RA' and spkr in all_nodes[n]['speaker']]
                    if verbose:
                        print(f"\t\t\t\tRA = {ra} \t I = {i}")
                        print(f"Edges in to RA")
                        print(f"\t\t\t\tRAs from {i}: ", ra_nodes_from_i)

                    # If there's more than one RA with this premise, then this RA is part of a divergent arg
                    # this is all the info we need about this RA, can move to next one
                    if len(ra_nodes_from_i) > 1:
                        ra_divergent[spkr]['ra_divergent_count'] += 1
                        break
            ra_divergent[spkr]['ra_divergent_perc'] = ra_divergent[spkr]['ra_divergent_count']/len(spkr_ra_all)
    else:
        ra_divergent['ra_divergent_count'] = 0

        if len(ra_nodes) == 0:
            ra_divergent['ra_divergent_perc'] = 0.0
            return ra_divergent


        for ra in ra_nodes:
            # get premise I-node of the relation
            i_node_to_ra = [n for n in all_nodes if ra in all_nodes[n]['eout'] and all_nodes[n]['type'] == 'I']

            if verbose:
                print(f"\t\tChecking RA {ra}")
                print(f"\t\t\tI-nodes to RA: ", i_node_to_ra)

            for i in i_node_to_ra:
                # Get all outgoing RAs of the premise
                ra_nodes_from_i = [n for n in all_nodes if i in all_nodes[n]['ein'] and all_nodes[n]['type'] == 'RA']
                if verbose:
                    print(f"\t\t\t\tRA = {ra} \t I = {i}")
                    print(f"Edges in to RA")
                    print(f"\t\t\t\tRAs from {i}: ", ra_nodes_from_i)

                # If there's more than one RA with this I-node as premise, then this RA is part of a divergent arg
                # this is all the info we need about this RA, can move to next one
                if len(ra_nodes_from_i) > 1:
                    ra_divergent['ra_divergent_count'] += 1
                    break
        ra_divergent['ra_divergent_perc'] = ra_divergent['ra_divergent_count']/len(ra_nodes)
    
    return ra_divergent


def ra_in_linked(xaif, speaker=False, verbose=False):
    if 'AIF' in xaif.keys():
        all_nodes, said = ova3.xaif_preanalytic_info_collection(xaif)
    else:
        all_nodes, said = ova2.xaif_preanalytic_info_collection(xaif)
    ra_nodes = [n for n in all_nodes if all_nodes[n]['type'] == 'RA']


    ra_linked = {}

    if speaker:
        for spkr in said:
            ra_linked[spkr] = {}
            spkr_ra_all = [n for n in ra_nodes if spkr in all_nodes[n]['speaker']]

            ra_linked[spkr]['ra_linked_count'] = 0

            if len(spkr_ra_all) == 0:
                ra_linked[spkr]['ra_linked_perc'] = 0.0
                continue

            for ra in spkr_ra_all:
                i_to_ra = [n for n in all_nodes[ra]['ein'] if all_nodes[n]['type'] == 'I']
                
                if len(i_to_ra) > 1:
                    ra_linked[spkr]['ra_linked_count'] += 1
            
                ra_linked[spkr]['ra_linked_perc'] = ra_linked[spkr]['ra_linked_count']/len(spkr_ra_all)


    else:
        ra_linked['ra_linked_count'] = 0

        if len(ra_nodes) == 0:
            ra_linked['ra_linked_perc'] = 0.0
            return ra_linked

        for ra in ra_nodes:
            i_to_ra = [n for n in all_nodes[ra]['ein'] if all_nodes[n]['type'] == 'I']
            
            if len(i_to_ra) > 1:
                ra_linked['ra_linked_count'] += 1
        
            ra_linked['ra_linked_perc'] = ra_linked['ra_linked_count']/len(ra_nodes)
    return ra_linked


def conflict_self(xaif, debug=False):
    if 'AIF' in xaif.keys():
        all_nodes, said = ova3.xaif_preanalytic_info_collection(xaif)
    else:
        all_nodes, said = ova2.xaif_preanalytic_info_collection(xaif)
    
    conflict_count = {}
    ca_nodes = [n for n in all_nodes if all_nodes[n]['type'] == 'CA']

    for spkr in said:
        conflict_count[spkr]= {'self_conflicts': 0}
        spkr_ca_all = [n for n in ca_nodes if spkr in all_nodes[n]['speaker']]
        
        for ca in spkr_ca_all:
            # Get node targeted by the CA
            ca_target = [n for n in all_nodes if n in all_nodes[ca]['eout']][0]
            target_speaker = all_nodes[all_nodes[ca_target]['introby'][0]]['speaker'][0]
            if target_speaker == spkr:
                conflict_count[spkr]['self_conflicts'] += 1
    
    return conflict_count


def circular_args(xaif):
    pass


def ca_undercut(xaif, speaker=False, verbose=False):
    all_nodes, said = ova3.xaif_preanalytic_info_collection(xaif)

    ca_nodes = [n for n in all_nodes if all_nodes[n]['type'] == 'CA']
    ra_nodes = [n for n in all_nodes if all_nodes[n]['type'] == 'RA']

    undercut_count = {}
    if speaker:
        for spkr in said:
            undercut_count[spkr] = {}
            spkr_ca_nodes = [n for n in ca_nodes if spkr in all_nodes[n]['speaker']]
            undercutting = [n for n in spkr_ca_nodes if set(all_nodes[n]['eout']).intersection(ra_nodes)]
            undercut_count[spkr]['undercut_count'] = len(undercutting)

    else:    
        # is an undercutter if its edges out include an edge to an RA
        undercutting = [n for n in ca_nodes if set(all_nodes[n]['eout']).intersection(ra_nodes)]
        undercut_count['undercut_count'] = len(undercutting)
    
    return undercut_count


def ca_rebut(xaif, speaker=False, verbose=False, skip_altgive=False):
    all_nodes, said = ova3.xaif_preanalytic_info_collection(xaif)

    ca_nodes = [n for n in all_nodes if all_nodes[n]['type'] == 'CA']
    if skip_altgive:
        alt_give_yas = [n for n in all_nodes if all_nodes[n]['text'] == 'Alternative Giving' and all_nodes[n]['type'] == 'YA']
        ca_nodes = [n for n in ca_nodes if not set(all_nodes[n]['ein']).intersection(set(alt_give_yas))]

    i_nodes = [n for n in all_nodes if all_nodes[n]['type'] == 'I']

    rebut_count = {}
    if speaker:
        for spkr in said:
            rebut_count[spkr] = {}
            spkr_ca_nodes = [n for n in ca_nodes if spkr in all_nodes[n]['speaker']]
            rebutting = [n for n in spkr_ca_nodes if set(all_nodes[n]['eout']).intersection(i_nodes)]
            rebut_count[spkr]['rebut_count'] = len(rebutting)

    else:    
        # is a rebutter if its edges out include an edge to an I
        rebutting = [n for n in ca_nodes if set(all_nodes[n]['eout']).intersection(i_nodes)]
        rebut_count['rebut_count'] = len(rebutting)
    
    return rebut_count




#####################
# Breadth and depth #
#####################

def initial_arg(xa_id, seen_xas, all_nodes, speaker=False, rel_type='RA', skip_altgive=False, verbose=False):
    seen_xas = seen_xas + [xa_id]
    
    if rel_type == 'CA' and skip_altgive:
        ca_nodes = [n for n in all_nodes if all_nodes[n]['type'] == 'CA']
        alt_give_yas = [n for n in all_nodes if all_nodes[n]['text'] == 'Alternative Giving' and all_nodes[n]['type'] == 'YA']
        cut_cas = [n for n in ca_nodes if set(all_nodes[n]['ein']).intersection(set(alt_give_yas))]

    initial_args = []

    if speaker:
        spkr = all_nodes[xa_id]['speaker'][0]
        spkr_xa_nodes = [n for n in all_nodes if all_nodes[n]['type'] == rel_type and spkr in all_nodes[n]['speaker'] and n not in seen_xas]
        if rel_type == 'CA' and skip_altgive:
            spkr_xa_nodes = [n for n in spkr_xa_nodes if n not in cut_cas]
    else:
        xa_nodes = [n for n in all_nodes if all_nodes[n]['type'] == rel_type and n not in seen_xas]
        if rel_type == 'CA' and skip_altgive:
            xa_nodes = [n for n in xa_nodes if n not in cut_cas]
    
    if verbose:
        print(f'Looking for earliest in chain from {rel_type} {xa_id}')
        print(f'Have seen the following {rel_type}s so far: ', seen_xas)
        if speaker:
            print(f'Associated speaker is {spkr}')
            print(f'Unseen {rel_type}s associated with this speaker: ', spkr_xa_nodes)
            print(f'All nodes saidby this speaker: ', [n for n in all_nodes if spkr in all_nodes[n]['speaker']])
            print(f'{rel_type} nodes saidby this speaker: ', [n for n in all_nodes if spkr in all_nodes[n]['speaker'] and all_nodes[n]['type'] == rel_type])
        

    # I-nodes premise(s) to this XA
    xa_premises = [n for n in all_nodes if xa_id in all_nodes[n]['eout'] and all_nodes[n]['type'] == 'I']

    if verbose:
        print()
        print(f"Premises for this {rel_type}: ", xa_premises)

    for i_premise in xa_premises:
        if speaker:
            xas_in = [n for n in spkr_xa_nodes if all_nodes[n]['nodeID'] in all_nodes[i_premise]['ein'] and all_nodes[n]['type'] == rel_type]
        else:
            xas_in = [n for n in xa_nodes if all_nodes[n]['nodeID'] in all_nodes[i_premise]['ein'] and all_nodes[n]['type'] == rel_type]

        if skip_altgive:
            xas_in = [n for n in xas_in if n not in cut_cas]

        if verbose:
            print(f"Checking premise {i_premise}: {all_nodes[i_premise]['text']}")
            print(f"\t{i_premise} has the following {rel_type}s in: ", xas_in)

        if len(xas_in) == 0:
            initial_args.append(xa_id)
        else:
            for xa in xas_in:
                initial_args = initial_args + initial_arg(xa, seen_xas, all_nodes, speaker=speaker, rel_type=rel_type, skip_altgive=skip_altgive)
        if verbose:
            print()

    # Avoid repeats
    return list(set(initial_args))



def path_lens_from_arg(xa_id, seen_xas, all_nodes, rel_type='RA', speaker=False, verbose=False):
    if verbose:
        print(f"\nChecking path lengths from {xa_id}")
    seen_xas = seen_xas + [xa_id]
    if verbose:
        print(f"Have now seen ", seen_xas)
    
    
    if speaker:
        target_spkr = all_nodes[xa_id]['speaker'][0]
    i_node_concls = [n for n in all_nodes if xa_id in all_nodes[n]['ein'] 
                     and all_nodes[n]['type'] == 'I']
    
    path_list = []

    if verbose:
        print("Targeting I-node(s)", i_node_concls)

    for i_concl in i_node_concls:
        if speaker:
            xas_out = [n for n in all_nodes if i_concl in all_nodes[n]['ein'] 
                    and all_nodes[n]['type'] == rel_type 
                    and all_nodes[n]['speaker'][0] == target_spkr]
        else:
            xas_out = [n for n in all_nodes if i_concl in all_nodes[n]['ein'] 
                    and all_nodes[n]['type'] == rel_type]
        if verbose:
            print(f"\tI-node {i_concl} has outgoing {rel_type}-nodes ", xas_out)
            

        if len(xas_out) == 0:   
            path_list.append(1)
        else:
            for xa_out in xas_out:
                # Avoid a loop: don't re-traverse anything traversed before
                if xa_out in seen_xas:
                    if verbose:
                        print(f"Have already seen {rel_type} {xa_out}, not following that path")
                    # But not just that it's been seen, this is therefore the end of a path! so should be recorded
                    path_list.append(1)

                    continue
                if verbose: 
                    print(f"\tFollowing path for {xa_out}")
                path_list = path_list + [x + 1 for x in path_lens_from_arg(xa_out, seen_xas, all_nodes, speaker=speaker, rel_type=rel_type, verbose=verbose)]
                # path_list = path_list + path_lens_from_arg(ra_out, all_nodes).apply(lambda x: x + 1)
    
    return path_list


def arg_depths(xaif, rel_type='RA', speaker=False, verbose=False, skip_altgive=False):
    if 'AIF' in xaif.keys():
        all_nodes, said = ova3.xaif_preanalytic_info_collection(xaif)
    else:
        all_nodes, said = ova2.xaif_preanalytic_info_collection(xaif)
    xa_nodes = [n for n in all_nodes if all_nodes[n]['type'] == rel_type]



    xa_depths = {}
    if rel_type == 'RA':
        depth_type = 'arg'
    elif rel_type == 'CA':
        depth_type = 'ca'

        # If using CAs and skipping alternative giving cases, remove any CAs from consideration which 
        # have an incoming edge from anAlternative Giving node
        if skip_altgive:
            alt_give_yas = [n for n in all_nodes if all_nodes[n]['text'] == 'Alternative Giving' and all_nodes[n]['type'] == 'YA']
            xa_nodes = [n for n in xa_nodes if not set(all_nodes[n]['ein']).intersection(set(alt_give_yas))]


    if speaker:
        # Per speaker
        for spkr in said:
            if verbose:
                print("Checking speaker ", spkr)
            xa_depths[spkr] = {f'{depth_type}_depths': []}

            # Get starters (but this ignores any circular args)
            spkr_starts = []
            
            # All RAs anchored by spkr's locs
            spkr_xa_all = [n for n in xa_nodes if spkr in all_nodes[n]['speaker']]

            # get all starts (this is ugly/inefficient but oh well)
            for xa in spkr_xa_all:
                spkr_starts += initial_arg(xa, [], all_nodes, speaker=speaker, verbose=verbose, rel_type=rel_type, skip_altgive=skip_altgive)
            spkr_starts = list(set(spkr_starts))
            
            # Follow each argument path from the first RAs in each path
            for starter_arg in spkr_starts:
                if verbose:
                    print("Looking for argument ", starter_arg)
                xa_depths[spkr][f'{depth_type}_depths'] = xa_depths[spkr][f'{depth_type}_depths'] + path_lens_from_arg(starter_arg, [], all_nodes, speaker=speaker, verbose=verbose)

    else:
        xa_depths[f'{depth_type}_depths'] = []
        # Get starters
        starts = []
        for xa in xa_nodes:
            starts += initial_arg(xa, [], all_nodes, speaker=speaker, verbose=verbose, rel_type=rel_type)
        starts = list(set(starts))

        # Follow each argument path from the first RAs in each path
        for starter_arg in starts:
            if verbose:
                print("Looking for argument ", starter_arg)
            xa_depths[f'{depth_type}_depths'] = xa_depths[f'{depth_type}_depths'] + path_lens_from_arg(starter_arg, [], all_nodes, rel_type=rel_type, speaker=speaker, verbose=verbose)

    return xa_depths

def max_ra_chain(xaif, speaker=False, verbose=False):
    all_depths = arg_depths(xaif, speaker=speaker, verbose=verbose)
    if speaker:
        max_ra = {}
        for spkr in all_depths:
            max_ra[spkr] = {}
            if len(all_depths[spkr]['arg_depths']) > 0:
                max_ra[spkr]['max_ra_chain'] = max(all_depths[spkr]['arg_depths'])
            else:
                max_ra[spkr]['max_ra_chain'] = 0
        return max_ra 
    else:
        if len(all_depths['arg_depths']) > 0:
            return {'max_ra_chain': max(all_depths['arg_depths'])}
        else:
            return {'max_ra_chain': 0}



def max_ca_chain(xaif, speaker=False, verbose=False):
    all_depths = arg_depths(xaif, speaker=speaker, verbose=verbose, rel_type='CA')
    if speaker:
        max_ca = {}
        for spkr in all_depths:
            max_ca[spkr] = {}
            if len(all_depths[spkr]['ca_depths']) > 0:
                max_ca[spkr]['max_ca_chain'] = max(all_depths[spkr]['ca_depths'])
            else:
                max_ca[spkr]['max_ca_chain'] = 0
        return max_ca 
    else:
        if len(all_depths['ca_depths']) > 0:
            return {'max_ca_chain': max(all_depths['ca_depths'])}
        else:
            return {'max_ca_chain': 0}


def arg_breadths(xaif, speaker=False, verbose=False):
    if 'AIF' in xaif.keys():
        all_nodes, said = ova3.xaif_preanalytic_info_collection(xaif)
    else:
        all_nodes, said = ova2.xaif_preanalytic_info_collection(xaif)
    ra_nodes = [n for n in all_nodes if all_nodes[n]['type'] == 'RA']

    ra_breadths = {}

    if speaker:
        for spkr in said:
            if verbose:
                print("Checking speaker", spkr)
            ra_breadths[spkr] = {'arg_breadths': []}

            # Get all I-nodes supported by speaker
            spkr_ra_all = [n for n in ra_nodes if spkr in all_nodes[n]['speaker']]
            
            # I-nodes with an incoming edge from an RA associated with the speaker
            spkr_suppd_i_nodes = [n for n in all_nodes 
                                if all_nodes[n]['type'] == 'I'
                                and set(spkr_ra_all) & set(all_nodes[n]['ein'])]
            
            if verbose:
                print(f"\tRA nodes:", spkr_ra_all)
                print(f"\tSupports the following I-nodes: ")
                for i in spkr_suppd_i_nodes:
                    print(f"\t\t{i}: {all_nodes[i]['text']}")


            # Now have all speaker's RAs and all I-nodes they support
            for supported in spkr_suppd_i_nodes:
                supp_count = 0
                
                # All incoming RAs from speaker
                support_relations = [n for n in all_nodes[supported]['ein'] if n in spkr_ra_all]
                if verbose:
                    print(f"\tI-node {supported} supported by RA", *support_relations)
                
                # Count I-nodes to RA that are attributed to the speaker (for handling linked arguments)
                for ra in support_relations:
                    all_i_nodes_in = [n for n in all_nodes[ra]['ein'] if all_nodes[n]['type'] == 'I']

                    # I-node was introduced by an L-node attributed to the speaker
                    i_nodes_in = [n for n in all_i_nodes_in if set(all_nodes[n]['introby']) & set(said[spkr])]
                    supp_count += len(i_nodes_in)

                    if verbose:
                        print(f"\t\t{spkr} I-nodes incoming to RA {ra}:", i_nodes_in)
                        print(f"\t\tSupport count from this RA-node is {len(i_nodes_in)}, total support count for I-node so far is {supp_count}")
                
                ra_breadths[spkr]['arg_breadths'].append(supp_count)
    else:
        ra_breadths['arg_breadths'] = []
        # Get all supported I-nodes
        suppd_i_nodes = [n for n in all_nodes 
                         if all_nodes[n]['type'] == 'I' 
                         and set(ra_nodes) & set(all_nodes[n]['ein'])]
        
        if verbose:
            print(f"\tRA nodes:", ra_nodes)
            print(f"\tSupports the following I-nodes: ")
            for i in suppd_i_nodes:
                print(f"\t\t{i}: {all_nodes[i]['text']}")
        
        for supported in suppd_i_nodes:
            supp_count = 0
                
            # All incoming RAs from speaker
            support_relations = [n for n in all_nodes[supported]['ein'] if n in ra_nodes]
            if verbose:
                print(f"\tI-node {supported} supported by RA", *support_relations)
            
            # Count I-nodes to RA that are attributed to the speaker (for handling linked arguments)
            for ra in support_relations:
                all_i_nodes_in = [n for n in all_nodes[ra]['ein'] if all_nodes[n]['type'] == 'I']

                # Map-level: doesn't matter where support is from
                supp_count += len(all_i_nodes_in)

                if verbose:
                    print(f"\t\tI-nodes incoming to RA {ra}:", all_i_nodes_in)
                    print(f"\t\tSupport count from this RA-node is {len(i_nodes_in)}, total support count for I-node so far is {supp_count}")
            
            ra_breadths['arg_breadths'].append(supp_count)
    
    return ra_breadths


def avg_arg_depths(xaif, speaker=False, verbose=False):
    depths = arg_depths(xaif, speaker=speaker)
    if speaker:
        for spkr in depths:
            try:
                depths[spkr]['avg_arg_depth'] = sum(depths[spkr]['arg_depths'])/len(depths[spkr]['arg_depths'])
            except ZeroDivisionError:
                depths[spkr]['avg_arg_depth'] = 0
    else:
        if len(depths['arg_depths']) == 0:
                depths['avg_arg_depth'] = 0
        else:
            depths['avg_arg_depth'] = sum(depths['arg_depths'])/len(depths['arg_depths'])

    return depths


def avg_arg_breadths(xaif, speaker=False):
    breadths = arg_breadths(xaif, speaker=speaker)
    if speaker:
        for spkr in breadths:
            try: 
                breadths[spkr]['avg_arg_breadth'] = sum(breadths[spkr]['arg_breadths'])/len(breadths[spkr]['arg_breadths'])
            except ZeroDivisionError:
                breadths[spkr]['avg_arg_breadth'] = 0
    else:
        if len(breadths['arg_breadths']) == 0:
            breadths['avg_arg_breadth'] = 0
        else:
            breadths['avg_arg_breadth'] = sum(breadths['arg_breadths'])/len(breadths['arg_breadths'])
    return breadths

######################
# Illocutionary acts #
######################


# Report illocutionary acts used to complete the second part of an argument
# Assumes this can be read from the locution which the Arguing-anchoring TA has an edge towards
def arg_intros(xaif, verbose=False):
    if 'AIF' in xaif.keys():
        all_nodes, said = ova3.xaif_preanalytic_info_collection(xaif)
    else:
        all_nodes, said = ova2.xaif_preanalytic_info_collection(xaif)
    
    # Ways of introducing the proposition associated with support, conflict and rephrase

    intro_yas = {}
    ra_nodes = [n for n in all_nodes if all_nodes[n]['type'] == 'RA']
    ca_nodes = [n for n in all_nodes if all_nodes[n]['type'] == 'CA']
    ma_nodes = [n for n in all_nodes if all_nodes[n]['type'] == 'MA']
    for spkr in said:
        intro_yas[spkr] = {'arg_intros': {
            'RA': {},
            'CA': {},
            'MA': {}
        }}
        
        spkr_ra_all = [n for n in ra_nodes if spkr in all_nodes[n]['speaker']]
        spkr_ca_all = [n for n in ca_nodes if spkr in all_nodes[n]['speaker']]
        spkr_ma_all = [n for n in ma_nodes if spkr in all_nodes[n]['speaker']]

        for relation_id_list, relation_txt in zip([spkr_ra_all, spkr_ca_all, spkr_ma_all], ['RA', 'CA', 'MA']):
            for relation_id in relation_id_list:
                l_node = arg_rel_lnode(relation_id, all_nodes)

                ya_type = [all_nodes[n]['text'] for n in all_nodes if all_nodes[n]['type'] == 'YA' 
                    and l_node in all_nodes[n]['ein']]
                
                if verbose:
                    print(f"Checking for {relation_txt} relation {relation_id}")
                    print(f"Associated l-node found to be: {l_node}")
                    print(f"\t{all_nodes[l_node]['text']}")
                    print('\t', *ya_type)

                
                if len(ya_type) == 0:
                    if verbose:
                        print('no YAs found!')
                        print(f'Relation: {relation_id} in {relation_txt} list {relation_id_list}')
                elif len(ya_type) > 1:
                    if verbose:
                        print('too many YAs found!', ya_type)
                        print(f'Relation: {relation_id} in {relation_txt} list {relation_id_list}')
                else:
                    if ya_type[0] not in intro_yas[spkr]['arg_intros'][relation_txt]:
                        intro_yas[spkr]['arg_intros'][relation_txt][ya_type[0]] = 1
                    else: 
                        intro_yas[spkr]['arg_intros'][relation_txt][ya_type[0]] += 1

            if verbose:
                print()
    
    return intro_yas



#########
# INTER #
#########


# No. of inferences which have as direct earlier component a proposition originating with another speaker
def direct_args_from_others(xaif, debug=False):
    if 'AIF' in xaif.keys():
        all_nodes, said = ova3.xaif_preanalytic_info_collection(xaif)
    else:
        all_nodes, said = ova2.xaif_preanalytic_info_collection(xaif)
    
    other_arg_count = {}
    ra_nodes = [n for n in all_nodes if all_nodes[n]['type'] == 'RA']


    for spkr in said:
        other_arg_count[spkr] = {'direct_args_from_others': 0}
        spkr_ra_all = [n for n in ra_nodes if spkr in all_nodes[n]['speaker']]

        for ra in spkr_ra_all:
            # Identify the I-node used to create the argument
            l = arg_rel_lnode(ra, all_nodes)
            ya_node_out = [n for n in all_nodes if all_nodes[n]['type'] == 'YA' and l in all_nodes[n]['ein']]

            if debug:
                print(f"Tried L-node {l}:", all_nodes[l]['text'])
                print(f"YA nodes found were:", ya_node_out)
            
            for ya in ya_node_out:
                connected_i_nodes = [n for n in all_nodes if all_nodes[n]['type'] == 'I' and ya in all_nodes[n]['ein'] 
                                    and (ra in all_nodes[n]['ein'] or ra in all_nodes[n]['eout'])]
                if len(connected_i_nodes) != 0:
                    argmaker_i = connected_i_nodes[0]
                    break
                else:
                    argmaker_i = ''
                    if debug:
                        print(f"No connected I-nodes found for RA {ra}")

            
            # Get I-nodes connected to RA node in the other direction
            if argmaker_i in all_nodes[ra]['ein']:
                checkable_i_nodes = [n for n in all_nodes if n in all_nodes[ra]['eout'] 
                                     and all_nodes[n]['type'] == 'I']
            else:
                checkable_i_nodes = [n for n in all_nodes if n in all_nodes[ra]['ein'] 
                                     and all_nodes[n]['type'] == 'I']


            # Check if any other-speaker prop was used
            # Get speakers of introductory locutions of all the relevant I-nodes
            loc_lists = [all_nodes[i]['introby'] for i in checkable_i_nodes]
            locs = [s for l_list in loc_lists for s in l_list]
            loc_speakers = [all_nodes[l]['speaker'][0] for l in locs]

            if debug:
                print("Locs: ", locs)
                print("Loc speakers: ", loc_speakers)
                for l in locs:
                    print(f"\t{l}: {all_nodes[l]['text']}")
                
                print("Arguer:", spkr)
                print("Other relevant speakers:", *set(loc_speakers))
            
            # Set of speakers for the other nodes contains more speakers than the RA-creator
            if set(loc_speakers) != {spkr}:
                other_arg_count[spkr]['direct_args_from_others'] += 1
    
    return other_arg_count


# Number of arguments by speaker where one (or more) of the propositions attributed to the arguing speaker 
# is a rephrase of a proposition introduced by another speaker
def indirect_args_from_others(xaif, debug=False):
    if 'AIF' in xaif.keys():
        all_nodes, said = ova3.xaif_preanalytic_info_collection(xaif)
    else:
        all_nodes, said = ova2.xaif_preanalytic_info_collection(xaif)
    
    other_arg_count = {}
    ra_nodes = [n for n in all_nodes if all_nodes[n]['type'] == 'RA']

    # Get all props associated with speaker that are connected to the RA
    for spkr in said:
        if debug:
            print(f"Checking for speaker {spkr}")
        other_arg_count[spkr] = {'indirect_args_from_others': 0}
        
        spkr_ra_all = [n for n in ra_nodes if spkr in all_nodes[n]['speaker']]

        for ra in spkr_ra_all:
            # Use a boolean to avoid over-counting if a linked arg includes multiple rephrased props
            other_found = False

            # Get all connected props
            props_in_arg = props_linked_to_rel(ra, all_nodes)
            prop_spkrs = [(p, i_node_introducer(p, all_nodes)) for p in props_in_arg]

            if debug:
                print(f"\tChecking RA {ra}:")
                print(f"\t\tProps:", props_in_arg)

            # Get props from arguing speaker
            own_props = []
            for prop, p_spkr in prop_spkrs:
                if debug:
                    print(f"\t\t{prop}, {p_spkr}: {all_nodes[prop]['text']}")
                if spkr == p_spkr:
                    own_props.append(prop)
            
            # Chech if any are rephrases of another speaker's prop
            for prop in own_props:
                ma_nodes = [n for n in all_nodes if all_nodes[n]['type'] == 'MA' and prop in all_nodes[n]['ein']]

                if debug:
                    if len(ma_nodes) == 0:
                        print(f"\t\tNo MA nodes connected to I-node {prop}")
                    else:
                        print(f"\t\tMA nodes connected to I-node {prop}:", ma_nodes)

                for ma in ma_nodes:
                    # original prop being rephrased
                    orig_prop = [n for n in all_nodes if all_nodes[n]['type'] == 'I' and n in all_nodes[ma]['eout']][0]
                    if i_node_introducer(orig_prop, all_nodes) != spkr:
                        other_arg_count[spkr]['indirect_args_from_others'] += 1
                        other_found = True
                    break
                if other_found:
                    break
    
    return other_arg_count


# Count of conflicts each speaker creates with content from another speaker
def dir_conflict_others(xaif, debug=False):
    if 'AIF' in xaif.keys():
        all_nodes, said = ova3.xaif_preanalytic_info_collection(xaif)
    else:
        all_nodes, said = ova2.xaif_preanalytic_info_collection(xaif)
    
    other_conflict_count = {}
    ca_nodes = [n for n in all_nodes if all_nodes[n]['type'] == 'CA']

    for spkr in said:
        if debug:
            print(f"Checking conflicts created by {spkr}")
        other_conflict_count[spkr] = {'direct_other_conflicts': 0}
        spkr_ca_all = [n for n in ca_nodes if spkr in all_nodes[n]['speaker']]
        
        for ca in spkr_ca_all:
            # Get node targeted by the CA
            ca_target = [n for n in all_nodes if n in all_nodes[ca]['eout']][0]
            target_orig_spkr = all_nodes[all_nodes[ca_target]['introby'][0]]['speaker'][0]
            if debug:
                    print(f"CA {ca} targets node {ca_target}")
                    print(f"\t{ca_target}, {target_orig_spkr}: {all_nodes[ca_target]['text']}")

            if target_orig_spkr != spkr:
                other_conflict_count[spkr]['direct_other_conflicts'] += 1
    
    return other_conflict_count

# Count of conflicts
def indir_conflict_others(xaif, debug=False):
    if 'AIF' in xaif.keys():
        all_nodes, said = ova3.xaif_preanalytic_info_collection(xaif)
    else:
        all_nodes, said = ova2.xaif_preanalytic_info_collection(xaif)
    
    other_conflict_count = {}
    ca_nodes = [n for n in all_nodes if all_nodes[n]['type'] == 'CA']

    for spkr in said:
        other_conflict_count[spkr] = {'indirect_other_conflicts': 0}
        spkr_ca_all = [n for n in ca_nodes if spkr in all_nodes[n]['speaker']]
        if debug:
            print(f"Checking conflicts created by {spkr}")
            print(f"{spkr} has CA nodes:", *spkr_ca_all)

        for ca in spkr_ca_all:
            other_found = False

            # Get node targeted by the CA
            ca_target = [n for n in all_nodes if n in all_nodes[ca]['eout']][0]
            target_loc = all_nodes[ca_target]['introby'][0]

            if debug:
                print(f"Checking conflict {ca} targeting I-node {ca_target}, {all_nodes[ca_target]['text']}")

            target_orig_spkr = all_nodes[target_loc]['speaker'][0]
            if debug:
                    print(f"CA {ca} targets node {ca_target}")
                    print(f"\t{ca_target}, {target_orig_spkr}: {all_nodes[ca_target]['text']}")

            # Target is from self: is it connected to another speaker?
            if target_orig_spkr == spkr:
                # 1) was it rephrased from a prop introduced by another speaker?
                # -> check for being descendent of rephrase from another
                originals = rephrased_by(ca_target, all_nodes)
                for orig in originals:
                    # This conflict invovles something traceable to another speaker
                    if i_node_introducer(orig, all_nodes) != spkr:
                        other_conflict_count[spkr]['indirect_other_conflicts'] += 1
                        other_found = True
                        break
                
                # Don't need to check anything else for this CA, skip rest of loop
                if other_found:
                    continue


                # 2) does it support or follow from a prop introduced by another speaker *earlier* in the convo?
                # -> check for connected RA nodes
                ras_to_target = [n for n in all_nodes 
                                if n in all_nodes[ca_target]['ein']
                                 and all_nodes[n]['type'] == 'RA']
                ras_from_target = [n for n in all_nodes 
                                if n in all_nodes[ca_target]['eout']
                                 and all_nodes[n]['type'] == 'RA']

                # Get props on other side of RA, check if chronologically earlier and from another speaker
                for ra in ras_to_target:
                    inodes_to_ra = [n for n in all_nodes[ra]['ein'] if all_nodes[n]['type'] == 'I']
                    for i in inodes_to_ra:
                        arg_loc = all_nodes[i]['introby'][0]
                        # Target is premise or conclusion for a proposition previously introduced by another speaker
                        if (all_nodes[arg_loc]['chron'] < all_nodes[target_loc]['chron']) and (all_nodes[arg_loc]['speaker'][0] != spkr):
                            other_conflict_count[spkr]['indirect_other_conflicts'] += 1
                            other_found = True
                            # no need to check other I-nodes
                            break
                    # no need to check any other RAs for the target prop
                    if other_found:
                        break
                
                # Don't need to check anything else for this CA, skip rest of loop
                if other_found:
                    continue

                for ra in ras_from_target:
                    inodes_from_ra = [n for n in all_nodes[ra]['eout'] if all_nodes[n]['type'] == 'I']
                    for i in inodes_from_ra:
                        arg_loc = all_nodes[i]['introby'][0]
                        # Target is premise or conclusion for a proposition previously introduced by another speaker
                        if (all_nodes[arg_loc]['chron'] < all_nodes[target_loc]['chron']) and (all_nodes[arg_loc]['speaker'][0] != spkr):
                            other_conflict_count[spkr]['indirect_other_conflicts'] += 1
                            other_found = True
                            # no need to check other I-nodes
                            break
                    # no need to check any other RAs for the target prop
                    if other_found:
                        break
    
    return other_conflict_count


# For each speaker, their action connected to a question
# YA anchored in the L-node and (if any) in the TA connecting to the question.
def follow_questions(xaif, debug=False):
    if 'AIF' in xaif.keys():
        all_nodes, said = ova3.xaif_preanalytic_info_collection(xaif)
    else:
        all_nodes, said = ova2.xaif_preanalytic_info_collection(xaif)

    # l_nodes = [n for n in all_nodes if all_nodes[n]['type'] == 'L']
    question_l_nodes = [n for n in all_nodes if all_nodes[n]['type'] == 'L' 
                        and node_anchors_text(n, 'Pure Questioning', all_nodes)]
    
    following_questions = {}
    for spkr in said:
        following_questions[spkr] = {}
    
    # Check what follows from each question
    for q_lnode in question_l_nodes:
        q_spkr = all_nodes[q_lnode]['speaker'][0]

        # Get TAs from the question made by a speaker other than the questioner
        ta_from_q = [n for n in all_nodes[q_lnode]['eout'] if all_nodes[n]['type'] == 'TA' and q_spkr not in all_nodes[n]['speaker']]
        l_from_q = []

        # Get YAs anchored in a from-question TA and L-nodes descending from a from-question TA
        for ta in ta_from_q:
            # Get any YA anchored by the TA
            yas = [n for n in all_nodes[ta]['eout'] if all_nodes[n]['type'] == 'YA']

            # Get descending L-nodes
            l_from_q = l_from_q + [n for n in all_nodes[ta]['eout'] if all_nodes[n]['type'] == 'L']

        
        # Get any YA anchored by the collected L-nodes
        for l in l_from_q:
            yas = yas + [n for n in all_nodes[l]['eout'] if all_nodes[n]['type'] == 'YA']
        
        # Attribute YAs to speakers
        for ya in yas:
            spkr = all_nodes[ya]['speaker'][0] # ASSUMPTION
            ya_text = all_nodes[ya]['text']
            # Check keys
            if ya_text not in following_questions[spkr].keys():
                following_questions[spkr][ya_text] = 1
            else:
                following_questions[spkr][ya_text] += 1
    
    return following_questions
            


def conflict_support(xaif, debug=False):
    if 'AIF' in xaif.keys():
        all_nodes, said = ova3.xaif_preanalytic_info_collection(xaif)
    else:
        all_nodes, said = ova2.xaif_preanalytic_info_collection(xaif)

    conflict_support = {}
    for spkr in said:
        conflict_support[spkr] = 0


    # Create tuples with IDs of i-nodes and ca node in conflicts
    conflict_tuples = []
    for ca in [c for c in all_nodes if all_nodes[c]['type'] == 'CA']:
        conflict_tuples = conflict_tuples + [([i for i in all_nodes if i in all_nodes[ca]['ein'] and all_nodes[i]['type'] == 'I'][0], 
                                            ca,
                                            [i for i in all_nodes if i in all_nodes[ca]['eout'] and all_nodes[i]['type'] == 'I'][0])]

    # Keep only those where the propositions involved originate with different speakers
    # (i.e. keep only cross-speaker conflicts)
    for tup in conflict_tuples:
        i_ca_premise_loc = all_nodes[tup[0]]['introby'][0]
        i_ca_premise_spkr = all_nodes[i_ca_premise_loc]['speaker'][0]

        i_ca_target_loc = all_nodes[tup[2]]['introby'][0]
        i_ca_target_spkr = all_nodes[i_ca_target_loc]['speaker'][0]

        # Internal conflict, irrelevant: move onto next CA
        if i_ca_premise_spkr == i_ca_target_spkr:
            continue
        
        # Look for supports for the attacking prop
        support = [n for n in all_nodes if all_nodes[n]['type'] == 'RA' and n in all_nodes[tup[0]]['ein']]

        for s in support:
            supp_spkr = all_nodes[s]['speaker'][0]

            # Support is from a speaker already involved in the conflict, irrelevant: move on to next support
            if supp_spkr == i_ca_premise_spkr:
                continue
            elif supp_spkr == i_ca_target_spkr:
                continue
            
            # Check support was created after the conflict was introduced
            supp_premises = [n for n in all_nodes[s]['ein'] if all_nodes[n]['type'] == 'I']
            for prem in supp_premises:
                supp_premise_loc = all_nodes[prem]['introby'][0]
                # if later than the attacking proposition, it's relevant
                if all_nodes[supp_premise_loc]['chron'] > all_nodes[i_ca_premise_loc]['chron']:
                    supp_premise_spkr = all_nodes[supp_premise_loc]['speaker'][0]
                    
                    # Support from a speaker other than the attacking one
                    # Record that speaker as supporting an attack
                    if supp_premise_spkr != i_ca_premise_spkr:
                        conflict_support[supp_premise_spkr] += 1
    
    return conflict_support
            

# Presence of agreement with or support for proposition introduced by another speaker 
# in same turn as a conflict with content from that speaker
# ! what about multiple CAs against the same turn?
# ! and attack+support from multipe speakers?
def face_protector(xaif, debug=False):
    if 'AIF' in xaif.keys():
        all_nodes, said = ova3.xaif_preanalytic_info_collection(xaif)
    else:
        all_nodes, said = ova2.xaif_preanalytic_info_collection(xaif)

    support_with_conflict = {}
    for spkr in said:
        support_with_conflict[spkr] = 0
    
    # Get conflicts
    conflicts = []
    for ca in [c for c in all_nodes if all_nodes[c]['type'] == 'CA']:
        conflicts = conflicts + [{
            'premise_i_id': [i for i in all_nodes if i in all_nodes[ca]['ein'] and all_nodes[i]['type'] == 'I'][0],
            'ca_id': ca,
            'target_i_id': [i for i in all_nodes if i in all_nodes[ca]['eout'] and all_nodes[i]['type'] == 'I'][0]
        }]

    # Keep only those CAs where the propositions involved originate with different speakers
    # (i.e. keep only cross-speaker conflicts)
    x_spkr_ca = []
    for con in conflicts:
        i_ca_premise_loc = all_nodes[con['premise_i_id']]['introby'][0]
        i_ca_premise_spkr = all_nodes[i_ca_premise_loc]['speaker'][0]

        i_ca_target_loc = all_nodes[con['target_i_id']]['introby'][0]
        i_ca_target_spkr = all_nodes[i_ca_target_loc]['speaker'][0]

        if i_ca_premise_spkr != i_ca_target_spkr:
            con['premise_loc_id'] = i_ca_premise_loc
            con['premise_spkr'] = i_ca_premise_spkr
            con['target_loc_id'] = i_ca_target_loc
            con['target_spkr'] = i_ca_target_spkr
            x_spkr_ca = x_spkr_ca + [con]
            
    if debug:
        print("X-spkr CAs: ", x_spkr_ca)

    # Get supports
    supports = []
    for ra in [n for n in all_nodes if all_nodes[n]['type'] == 'RA']:
        supports = supports + [{
            'premise_i_id': [i for i in all_nodes if i in all_nodes[ra]['ein'] and all_nodes[i]['type'] == 'I'][0],
            'ra_id': ra,
            'concl_i_id': [i for i in all_nodes if i in all_nodes[ra]['eout'] and all_nodes[i]['type'] == 'I'][0]
        }]
    
    # Keep only those RAs where the propositions involved originate with different speakers
    x_spkr_ra = []
    for supp in supports:
        i_ra_premise_loc = all_nodes[supp['premise_i_id']]['introby'][0]
        i_ra_premise_spkr = all_nodes[i_ra_premise_loc]['speaker'][0]

        i_ra_target_loc = all_nodes[supp['concl_i_id']]['introby'][0]
        i_ra_target_spkr = all_nodes[i_ra_target_loc]['speaker'][0]

        if i_ra_premise_spkr != i_ra_target_spkr:
            supp['premise_loc_id'] = i_ra_premise_loc
            supp['premise_spkr'] = i_ra_premise_spkr
            supp['concl_loc_id'] = i_ra_target_loc
            supp['concl_spkr'] = i_ra_target_spkr            
            x_spkr_ra = x_spkr_ra + [supp]

    if debug:
        print("X-spkr RAs: ", x_spkr_ra)

    # Get agreement
    x_spkr_agree = []
    for agr in [n for n in all_nodes if all_nodes[n]['text'] == 'Agreeing' and all_nodes[n]['type'] == 'YA']:
        agree_spkr = all_nodes[agr]['speaker'][0]
        agree_target = [n for n in all_nodes[agr]['eout']][0]

        # Need loc resulting from agreement, to do chrono comparisons
        agree_loc = ''
        for a in all_nodes[agr]['ein']:
            if all_nodes[a]['type'] == 'L':
                agree_loc = a
            elif all_nodes[a]['type'] == 'TA':
                l_nodes_out = [n for n in all_nodes if all_nodes[n]['type'] == 'L' and a in all_nodes[n]['ein']]
                # if len(l_nodes_out) == 1:
                # *** Assuming there'll be only outgoing L from the agreeing TA
                agree_loc = l_nodes_out[0]

        if all_nodes[agree_target]['type'] == 'I':
            agr_target_loc = all_nodes[agree_target]['introby'][0]
            agr_target_spkr = all_nodes[agr_target_loc]['speaker'][0]
        else:
            agr_target_spkr = all_nodes[agree_target]['speaker'][0]
        if agree_spkr != agr_target_spkr:
            x_spkr_agree = x_spkr_agree + [{
                'agree_id': agr,
                'agree_loc_id': agree_loc,
                'speaker': agree_spkr,
                'target_i_id': agree_target,
                'target_loc_id': agr_target_loc,
                'target_spkr': agr_target_spkr
            }]


    # Get turns
    all_locs = [all_nodes[n] for n in all_nodes if all_nodes[n]['type'] == 'L']
    ordered_locs = all_locs.copy()
    ordered_locs.sort(key=lambda n:n['chron'])
    
    turns = []
    prev_spkr = ordered_locs[0]['speaker'][0]
    turns = turns + [{'speaker': prev_spkr, 'locs': [ordered_locs[0]]}]
    turns[0]['locs']

    for l in ordered_locs[1:]:
        if l['speaker'][0] == prev_spkr:
            turns[-1]['locs'] = turns[-1]['locs'] + [l]
        else:
            prev_spkr = l['speaker'][0]
            turns = turns + [{'speaker':prev_spkr, 'locs': [l]}]
    
    for ca in x_spkr_ca:
        ca_target_chron = all_nodes[ca['target_loc_id']]['chron']
        ca_attack_chron = all_nodes[ca['premise_loc_id']]['chron']
        
        ca_target_turn = None
        ca_attack_turn = None
        for turn in turns:
            turn_span = [loc['chron'] for loc in turn['locs']]
            if ca_target_chron in turn_span:
                ca_target_turn = turn_span
            if ca_attack_chron in turn_span:
                ca_attack_turn = turn_span
            if ca_target_turn and ca_attack_turn:
                if debug:
                    print(f"CA target in turn ", ca_target_turn)
                    print(f"CA attack takes place in turn ", ca_attack_turn)
                break
        
        # Now have the turn with the target and the turn with the attack
        # Are there supports/agreements directed from the speaker creating the conflcit to content from the speaker of the attacked content?
        candidate_supps =   [supp for supp in x_spkr_ra if supp['premise_spkr'] == ca['premise_spkr']
                                and supp['concl_spkr'] == ca['target_spkr']]
        candidate_agrs = [agr for agr in x_spkr_agree if agr['target_spkr'] == ca['target_spkr']
                                and agr['speaker'] == ca['premise_spkr']]
        if debug:
            print(f"CA target speaker: {ca['target_spkr']}")
            print(f"CA premise speaker: {ca['premise_spkr']}")
            for supp in x_spkr_ra:
                print(f"RA premise speaker: {supp['premise_spkr']}")
                print(f"RA conclusion speaker: {supp['concl_spkr']}")

            print("Candidate supports:", candidate_supps)
            print("Candidate agreements:", candidate_agrs)


        # Do any of these support/agree during the same turn as the conflict is created?
        candidate_supps = [supp for supp in candidate_supps if all_nodes[supp['premise_loc_id']]['chron'] in ca_attack_turn]
        candidate_agrs = [agr for agr in candidate_agrs if all_nodes[agr['agree_loc_id']]['chron'] in ca_attack_turn]

        # If any cases found, mark a case of 'supporting while conflicting' for the speaker
        if len(candidate_agrs) > 0 or len(candidate_supps) > 0:
            support_with_conflict[ca['premise_spkr']] += 1
    
    return support_with_conflict


# Report the YAs used by speakers to introduce the premise of a CA relation against another speaker's content
def conflict_illocs(xaif, debug=False):
    if 'AIF' in xaif.keys():
        all_nodes, said = ova3.xaif_preanalytic_info_collection(xaif)
    else:
        all_nodes, said = ova2.xaif_preanalytic_info_collection(xaif)

    other_conflict_yas = {}
    for spkr in said:
        other_conflict_yas[spkr] = {}
    
    # Get conflicts
    conflicts = []
    for ca in [c for c in all_nodes if all_nodes[c]['type'] == 'CA']:
        conflicts = conflicts + [{
            'premise_i_id': [i for i in all_nodes if i in all_nodes[ca]['ein'] and all_nodes[i]['type'] == 'I'][0],
            'ca_id': ca,
            'target_i_id': [i for i in all_nodes if i in all_nodes[ca]['eout'] and all_nodes[i]['type'] == 'I'][0]
        }]

    # Get cross-speaker conflicts only
    x_spkr_ca = []
    for con in conflicts:
        i_ca_premise_loc = all_nodes[con['premise_i_id']]['introby'][0]
        i_ca_premise_spkr = all_nodes[i_ca_premise_loc]['speaker'][0]

        i_ca_target_loc = all_nodes[con['target_i_id']]['introby'][0]
        i_ca_target_spkr = all_nodes[i_ca_target_loc]['speaker'][0]

        if i_ca_premise_spkr != i_ca_target_spkr:
            con['premise_loc_id'] = i_ca_premise_loc
            con['premise_spkr'] = i_ca_premise_spkr
            con['target_loc_id'] = i_ca_target_loc
            con['target_spkr'] = i_ca_target_spkr
            x_spkr_ca = x_spkr_ca + [con]

    # ! Only one YA assumed/checked, this should be tighter
    for con in x_spkr_ca:
        ya_node_in = [all_nodes[n]['text'] for n in all_nodes 
                      if all_nodes[n]['type'] == 'YA' 
                      and con['premise_i_id'] in all_nodes[n]['eout']
                      and con['premise_loc_id'] in all_nodes[n]['ein']][0]
        
        if ya_node_in not in other_conflict_yas[con['premise_spkr']].keys():
            other_conflict_yas[con['premise_spkr']][ya_node_in] = 1
        else:
            other_conflict_yas[con['premise_spkr']][ya_node_in] += 1
    
    return other_conflict_yas


# From prev: proportion of YAs that express agreement
# same equation as before: # spkr x agreeing / # all spkr x YAs
def sycophancy(xaif, debug=False):
    if 'AIF' in xaif.keys():
        all_nodes, said = ova3.xaif_preanalytic_info_collection(xaif)
    else:
        all_nodes, said = ova2.xaif_preanalytic_info_collection(xaif)

    sycophancy = {}
    for spkr in said:
        spkr_yas = [all_nodes[n]['text'] for n in all_nodes 
                    if all_nodes[n]['type'] == 'YA'
                    and all_nodes[n]['speaker'][0] == spkr]
        
        if len(spkr_yas) == 0:
            sycophancy[spkr] = 0
        else:
            ya_counts = Counter(spkr_yas)
            sycophancy[spkr] = ya_counts['Agreeing']/len(spkr_yas)
    
    return sycophancy
        

# From prev (ish): proportion of locutions that introduce conflict
# edited to be cross-speaker conflicts only
def belligerence(xaif, debug=False):
    if 'AIF' in xaif.keys():
        all_nodes, said = ova3.xaif_preanalytic_info_collection(xaif)
    else:
        all_nodes, said = ova2.xaif_preanalytic_info_collection(xaif)

    # Get conflicts
    conflicts = []
    for ca in [c for c in all_nodes if all_nodes[c]['type'] == 'CA']:
        conflicts = conflicts + [{
            'premise_i_id': [i for i in all_nodes if i in all_nodes[ca]['ein'] and all_nodes[i]['type'] == 'I'][0],
            'ca_id': ca,
            'target_i_id': [i for i in all_nodes if i in all_nodes[ca]['eout'] and all_nodes[i]['type'] == 'I'][0]
        }]

    # Get cross-speaker conflicts only
    x_spkr_ca = []
    for con in conflicts:
        i_ca_premise_loc = all_nodes[con['premise_i_id']]['introby'][0]
        i_ca_premise_spkr = all_nodes[i_ca_premise_loc]['speaker'][0]

        i_ca_target_loc = all_nodes[con['target_i_id']]['introby'][0]
        i_ca_target_spkr = all_nodes[i_ca_target_loc]['speaker'][0]

        if i_ca_premise_spkr != i_ca_target_spkr:
            con['premise_loc_id'] = i_ca_premise_loc
            con['premise_spkr'] = i_ca_premise_spkr
            con['target_loc_id'] = i_ca_target_loc
            con['target_spkr'] = i_ca_target_spkr
            x_spkr_ca = x_spkr_ca + [con]


    belligerence = {}
    for spkr in said:
        spkr_locs = [n for n in all_nodes 
                     if all_nodes[n]['type'] == 'L' 
                     and all_nodes[n]['speaker'][0] == spkr]

        if len(spkr_locs) == 0:
            belligerence[spkr] = 0
        else:
            spkr_x_spkr_cas = [ca for ca in x_spkr_ca if ca['premise_spkr'] == spkr]
            belligerence[spkr] = len(spkr_x_spkr_cas) / len(spkr_locs)
    
    return belligerence


# From prev: "The proportion of questions to a participant that lead to substantive answers 
# (i.e. responses that $rephrase$ the underspecified content of questions)."
# -> proportion of TAs by a speaker from a PQ which anchor an MA
# This also catches actual rephrases of the question, which seems inappropriate... and doesn't control for x-spker-ness
# but this is just reimplementation of the orig
def responsiveness_sic(xaif, debug=False):
    if 'AIF' in xaif.keys():
        all_nodes, said = ova3.xaif_preanalytic_info_collection(xaif)
    else:
        all_nodes, said = ova2.xaif_preanalytic_info_collection(xaif)

    responsiveness = {}
    
    pq_yas = [n for n in all_nodes if all_nodes[n]['text'] == 'Pure Questioning' and all_nodes[n]['type'] == 'YA']
    question_locs = []
    for pq in pq_yas:
        question_locs = question_locs + [n for n in all_nodes[pq]['ein'] 
                                         if all_nodes[n]['type'] == 'L']

    for spkr in said:
        # TAs attributed to speaker which have an incoming edge from a pure-question-anchoring L-node 
        spkr_tas_from_q = [n for n in all_nodes 
                       if all_nodes[n]['type'] == 'TA'
                       and all_nodes[n]['speaker'][0] == spkr
                       and set(all_nodes[n]['ein']) & set(question_locs)]

        if debug:
            print(f"Checking for {spkr}")
            print(f"{len(spkr_tas_from_q)} TAs from pq")

        if len(spkr_tas_from_q) == 0:
            responsiveness[spkr] = 0
        else:
            # YAs anchored by one of those TAs
            poss_yas = [ya for ya in all_nodes 
                        if all_nodes[ya]['type'] == 'YA'
                        and set(all_nodes[ya]['ein']) & set(spkr_tas_from_q)]
            
            # MAs anchored by one of those YAs
            mas = [n for n in all_nodes
                        if set(all_nodes[n]['ein']) & set(poss_yas)
                        and all_nodes[n]['type'] == 'MA']

            responsiveness[spkr] = len(mas)/len(spkr_tas_from_q)

    return responsiveness


# As above but x-spkr TAs only, and only counting as relevant MAs anchored by Restating
def responsiveness(xaif, debug=False):
    if 'AIF' in xaif.keys():
        all_nodes, said = ova3.xaif_preanalytic_info_collection(xaif)
    else:
        all_nodes, said = ova2.xaif_preanalytic_info_collection(xaif)

    responsiveness = {}
    
    pq_yas = [n for n in all_nodes if all_nodes[n]['text'] == 'Pure Questioning' and all_nodes[n]['type'] == 'YA']
    question_locs = []
    for pq in pq_yas:
        question_locs = question_locs + [n for n in all_nodes[pq]['ein'] 
                                         if all_nodes[n]['type'] == 'L']
    # Get cross-speaker transitions only
    x_spkr_tas_from_q = []
    for q_loc in question_locs:
        x_spkr_tas_from_q = x_spkr_tas_from_q + [n for n in all_nodes
                                                 if all_nodes[n]['type'] == 'TA'
                                                 and n in all_nodes[q_loc]['eout']
                                                 and all_nodes[q_loc]['speaker'][0] != all_nodes[n]['speaker'][0]]
    
    for spkr in said:
        # TAs attributed to speaker which have an incoming edge from a pure-question-anchoring L-node 
        # spkr_tas_from_q = [n for n in all_nodes 
        #                if all_nodes[n]['type'] == 'TA'
        #                and all_nodes[n]['speaker'][0] == spkr
        #                and set(all_nodes[n]['ein']) & set(question_locs)]
        current_spkr_tas_from_q = [n for n in x_spkr_tas_from_q if all_nodes[n]['speaker'][0] == spkr]

        if debug:
            print(f"Checking for {spkr}")
            # print(f"{len(x_spkr_tas_from_q)} TAs from pq")

        if len(current_spkr_tas_from_q) == 0:
            responsiveness[spkr] = 0
        else:
            # non-restating YAs anchored by one of those TAs
            poss_yas = [ya for ya in all_nodes 
                        if all_nodes[ya]['type'] == 'YA'
                        and all_nodes[ya]['text'] != 'Restating'
                        and set(all_nodes[ya]['ein']) & set(current_spkr_tas_from_q)]
            
            # MAs anchored by one of those YAs
            mas = [n for n in all_nodes
                        if set(all_nodes[n]['ein']) & set(poss_yas)
                        and all_nodes[n]['type'] == 'MA']

            responsiveness[spkr] = len(mas)/len(current_spkr_tas_from_q)

    return responsiveness

########################
# Node-Level#
######################## 

########################
# Counts#
######################## 

#Returns each I node and its word count
def node_wc(xaif):
    wordcount = {}
    wordcount_list = []
    inodes = [n for n in xaif['AIF']['nodes'] if n['type'] == "I"]
    for n in inodes:
        # wordcount[n["nodeID"]] = len(n['text'].split())
        
        wordcount_list.append({"node id:": n['nodeID'], "word count:": len(n['text'].split())})
    wordcount = {"word_count": wordcount_list}
    return wordcount

#Returns count of I nodes with incoming RA
def supportedNodes(xaif):

    inodes = [n['nodeID'] for n in xaif['AIF']['nodes'] if n['type'] == "I"]
    inode_incoming = [e['fromID'] for e in xaif['AIF']['edges'] if e['toID'] in inodes]

    supportNodes = [n for n in xaif['AIF']['nodes'] if n['nodeID'] in inode_incoming and n['type'] == 'RA']
    return {"supported_nodes": len(supportNodes)}

#Returns number of I nodes with incoming CA
def attackedNodes(xaif):
    inodes = [n['nodeID'] for n in xaif['AIF']['nodes'] if n['type'] == "I"]
    inode_incoming = [e['fromID'] for e in xaif['AIF']['edges'] if e['toID'] in inodes]

    attackNodes = [n for n in xaif['AIF']['nodes'] if n['nodeID'] in inode_incoming and n['type'] == 'CA']
    
    return {"attacked_nodes": len(attackNodes)}

# Verb tags: https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
# NLTK tagger: https://www.nltk.org/book/ch05.html
#Returns a past present and base verb count for each inode
def nodeTenseScores(xaif):
    past = int(0)
    present = int(0)
    base = int(0)
    tense_list = []
    tense_count = {}
    inodes = [n for n in xaif['AIF']['nodes'] if n['type'] == "I"]
    for node in inodes:
        x = nltk.word_tokenize(node['text'])
        token_text = nltk.pos_tag(x)
        for z, tag in token_text:
            if tag == "VBD" or tag == "VBN":
                past += 1
            elif tag == "VBG" or tag == "VBP" or tag == "VBZ":
                present += 1
            elif tag == "VB":
                base += 1
        tense_list.append({"node id": node['nodeID'], "past": past, "present": present, "base": base})
    tense_count = {"node_tenses": tense_list}
    return tense_count

#Tutorial: https://wellsr.com/python/python-named-entity-recognition-with-nltk-and-spacy/
#Returns identified named entities with label and nodeid
def ner(xaif):
    ner_values = {}
    ner_list = []
    inodes = [n for n in xaif['AIF']['nodes'] if n['type'] == "I"]
    for node in inodes:
        doc = nlp(node['text'])
        for ent in doc.ents:
            ner_list.append({"node id": node['nodeID'], "text": ent.text, "label": ent.label_})
    ner_values = {"named entities": ner_list}
    return ner_values

#1 = positive -1 = negative
def sentiment(xaif):
    sia = SentimentIntensityAnalyzer()
    sent_list = []

    inodes = [n for n in xaif['AIF']['nodes'] if n['type'] == "I"]
    for node in inodes:
        node_sent = sia.polarity_scores(node['text'])
        sent_list.append({"node id": node['nodeID'], "sentiment": node_sent})
    
    return {"sentiment": sent_list}




def addForecastAccuracy(xaif):
    print(xaif['text'])
    try:
        match = re.search(r"Part ID: ?\d+", xaif['text'])
        match_span = match.span()
        id = match.string[8:match_span[1]]
        with open ('forecast750_brier.csv', 'r') as file:
            csvfile = csv.reader(file)
            for line in csvfile:
                if line[1] == id:
                    print("In IF")
                    if line[16] != "unknown":
                        return {"accuracy": line[16]}
                    else:
                        return {"accuracy": "unknown"}
    except:
        print("No Part ID found")
    
def addNodeOutcomes(xaif):
    match = re.search(r"Part ID: ?\d+", xaif['text'])
    match_span = match.span()
    id = match.string[8:match_span[1]]
    index = 0

    with open ('forecast750_brier.csv', 'r') as file:
        node_outcomes = []
        csvfile = csv.reader(file)
        for line in csvfile:
            if line[1] == id:
                probability_list = []
                items = line[9][1:-1].split(',')
                probability_list.extend(items)
                outcome_list = []
                items = line[11][1:-1].split(',')
                outcome_list.extend(items)
                inodes = [n for n in xaif['AIF']['nodes'] if n['type'] == "I"]
                for inode in inodes:
                    match = re.search(r"\, \d?\.\d+ \w+", inode['text'])
                    # print(match)
                    if match != None:
                        match_span = match.span()
                        print("match span")
                        print(match_span)
                        match_split = match.string[match_span[0]:match_span[1]].split()
                        probability = match_split[1]
                        if probability in probability_list:
                            list_index = probability_list.index(probability)
                            outcome = ''
                            if len(outcome_list) > 1:
                                outcome = outcome_list[list_index]
                            elif len(outcome_list) == 1:
                                if outcome_list[0] == 1 or outcome_list[0] == 0:
                                    if int(outcome_list[0]) == 0:
                                        if list_index == 0:
                                            outcome = 0
                                        elif list_index == 1:
                                            outcome = 1
                                    elif int(outcome_list[0]) == 1:
                                        if list_index == 0:
                                            outcome = 1
                                        elif list_index == 1:
                                            outcome = 0
                                    node_outcomes.append({"nodeID": inode['nodeID'], "outcome": outcome})
                                else:
                                    node_outcomes.append({"nodeID": inode['nodeID'], "outcome": "unknown"})

                        else:
                            print("no match :(")
        
        return {"outcomes" : node_outcomes}
    

def getHypSubgraphs(xaif):
    print("in get hyp subgraphs")
    print(xaif)

    subgraphs = []

    inode_ids = [n['nodeID'] for n in xaif['AIF']['nodes'] if n['type'] == "I"]
    print(inode_ids)
    nodetoinode = [e['fromID'] for e in xaif['AIF']['edges'] if e['toID'] in inode_ids]

    #Get YA "Hypothesising" nodes
    hyp_nodes = [n['nodeID'] for n in xaif['AIF']['nodes'] if n['nodeID'] in nodetoinode and n['text'] == "Hypothesising"]

    #Get inodes which are anchored in YA "Hypothesising"
    
    for hnode in hyp_nodes:
        hyp_inodes = []
        inode = [e['toID'] for e in xaif['AIF']['edges'] if e['fromID'] == hnode]
        hyp_inodes.append(inode[0])
        print("Inodes anchored in YA Hypothesising: " + str(hyp_inodes))
    
        #Get incoming nodes to Hypothesising YA-Anchored I-nodes
        s_nodes = []
        inodes1 = []
        for hyp_inode in hyp_inodes:
            #List of s nodes incoming to hypothesis option
            incoming_s_list = [e['fromID'] for e in xaif['AIF']['edges'] if e['toID'] == hyp_inode]
            #List of MA nodes which are incoming to hypothesis option
            found_node_l = [n['nodeID'] for n in xaif['AIF']['nodes'] if n['type'] == "MA" and n['nodeID'] in incoming_s_list]
            for n in found_node_l:
                s_nodes.append(n)
            inodel1 = [e['fromID'] for e in xaif['AIF']['edges'] if e['toID'] in s_nodes]
            for id in inodel1:
                for node in xaif['AIF']['nodes']:
                    if node['nodeID'] == id and node['type'] == "I":
                        inodes1.append(id)
        #I nodes with forecasts and probabilities
        print("Level 1 inodes: " + str(inodes1))

        #Loop through nodes connected to forecast I-node
        current_node = ''
        moreNodes = True
        subgraph_nodes = []
        subgraph_edges = []


        #Find all incoming nodes to current node
        #Loop through and find all incoming nodes again with each node in loop becoming the current node


        for nodeid in inodes1:
            node = [n for n in xaif['AIF']['nodes'] if n['nodeID'] == nodeid]
            current_node = node[0]
            next_nodes = []

            subgraph_nodes = []
            subgraph_edges = []
            
            subgraph_nodes.append(current_node)

            while moreNodes:
                # print("------------------------")
                # print("Current Node: ")
                # print(current_node)
                


                incoming_edges = [e['fromID'] for e in xaif['AIF']['edges'] if e['toID'] == current_node['nodeID']]
                #This is the RA nodes or any other incoming YA nodes
                # print("Incoming edges/RAs: ")
                # print(incoming_edges)
                if len(incoming_edges) == 0:
                    moreNodes == False
                    break
                
                incoming_non_YA = [n['nodeID'] for n in xaif['AIF']['nodes'] if n['nodeID'] in incoming_edges and n['type'] != "YA"]
                edges_to_append = [e for e in xaif['AIF']['edges'] if e['fromID'] in incoming_edges and e['fromID'] in incoming_non_YA]
                
                if edges_to_append != []:
                    subgraph_edges.append(edges_to_append)
                edges_to_append = []

                # print("edges to append 1:")
                # print([e for e in xaif['AIF']['edges'] if e['fromID'] in incoming_edges and e['fromID'] in incoming_non_YA])
                
                # print("Incoming non YA: ")
                # print(incoming_non_YA)

                #These are the s nodes with edge going to current node
                n_to_append = ([n for n in xaif['AIF']['nodes'] if n['nodeID'] in incoming_non_YA])
                for item in n_to_append:
                    subgraph_nodes.append(item)
                next_level_edges = [e for e in xaif['AIF']['edges'] if e['toID'] in incoming_non_YA]
                ids_from_ra = [e['fromID'] for e in xaif['AIF']['edges'] if e['toID'] in incoming_non_YA]
                # print("Next level edge from ids: ")
                # print(next_level_edges_ids)
                next_level_inode = [n for n in xaif['AIF']['nodes'] if n['nodeID'] in ids_from_ra and n['type'] == "I"]

                next_level_inode_ids = [n['nodeID'] for n in xaif['AIF']['nodes'] if n['nodeID'] in ids_from_ra and n['type'] == "I"]



                # print("edges to append 2:")
                # print([e for e in xaif['AIF']['edges'] if e['fromID'] in next_level_inode_ids and e['toID'] in incoming_non_YA])
                edges_to_append = [e for e in xaif['AIF']['edges'] if e['fromID'] in next_level_inode_ids and e['toID'] in incoming_non_YA]
                # edges_to_append_ids = [e['fromID'] for e in xaif['AIF']['edges'] if e['fromID'] in next_level_inode_ids and e['toID'] in incoming_non_YA]

                if edges_to_append != []:
                    for item in edges_to_append:
                        subgraph_edges.append(item)
                edges_to_append = []
                # print("edges to append 2:")
                # print([e for e in xaif['AIF']['edges'] if e['fromID'] in next_level_edges_ids and e['fromID'] in next_level_inode_ids])

                # print("------------------------")
                # print("Inodes next step down: ")
                # print(next_level_inode)
                # print("LEN: " + str(len(next_level_inode)))

                if len(next_level_inode) != 0:
                    for item in next_level_inode:
                        next_nodes.append(item)
                if len(next_nodes) != 0:
                    for item in next_level_inode:
                        subgraph_nodes.append(item)
                    # for edge in next_level_edges:
                    #     print("edges 3:")
                    #     print(edge)
                    #     subgraph_edges.append(edge)
                    current_node = next_nodes[0]
                    next_nodes.pop(0)
                else:
                    break
                    # for inode in next_level_inode:
                
                # print("End of loop status")
                # print("current node: ")
                # print(current_node)
                # print("next nodes: ")
                # print(next_nodes)

     

        # subgraph
        
        # subgraph.append(subgraph_nodes)
        # subgraph.append(subgraph_edges)
        # print("Subgraph: ")
        # print(subgraph)
        # print("subgraph_nodes")
        # print(subgraph_nodes)

        xaif_subgraph = {
            "AIF": {
                'nodes': subgraph_nodes,
                'edges': subgraph_edges
            }
        }
        # print(xaif_subgraph)

        subgraphs.append(xaif_subgraph)
    
    return subgraphs


def raCount(xaif):
    ra_nodes = [n for n in xaif['AIF']['nodes'] if n['type'] == 'RA']
    return {"ra_count" : len(ra_nodes)}

def caCount(xaif):
    ca_nodes = [n for n in xaif['AIF']['nodes'] if n['type'] == 'CA']
    return {"ca_count" : len(ca_nodes)}

def forecast_wc(xaif):
    wc_per_node = node_wc(xaif)
    print(wc_per_node)
    sum = 0
    for node in wc_per_node['word_count']:
        num = node.get("word count:")
        sum += num
    return({"word count" : sum})

                        

