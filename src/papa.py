import json
from . import xaif_toolbox as ova3
from . import analytics
import sys
import os

def all_analytics(xaif, node_level=False, speaker=False, forecast=False):
    xaif = ova3.ova2_to_ova3(xaif)

    # This involves redundancy currently bc all analytics are designed to call the info collection
    # so that they can be run individually with just the XAIF...


    # Speaker-level analytics
    if speaker:
        if 'AIF' in xaif.keys():
            wordcounts = ova3.spkr_wordcounts(xaif)
            spkr_analytic_list = []
                
        spkr_rel_counts = analytics.arg_relation_counts(xaif, speaker=True)

        spkr_analytic_list.append(wordcounts)
        spkr_analytic_list.append(analytics.direct_args_from_others(xaif))
        spkr_analytic_list.append(analytics.indirect_args_from_others(xaif))

        spkr_analytic_list.append(analytics.loc_counts(xaif, speaker=speaker))
        spkr_analytic_list.append(analytics.arg_word_densities(xaif, speaker=speaker))
        spkr_analytic_list.append(analytics.arg_loc_densities(xaif, speaker=speaker))
        
        spkr_analytic_list.append(analytics.arg_intros(xaif))
        spkr_analytic_list.append(analytics.concl_first_perc(xaif, speaker=speaker))
        spkr_analytic_list.append(analytics.avg_arg_breadths(xaif, speaker=speaker))
        spkr_analytic_list.append(analytics.avg_arg_depths(xaif, speaker=speaker))
        spkr_analytic_list.append(analytics.ra_in_serial(xaif, speaker=speaker))
        spkr_analytic_list.append(analytics.ra_in_convergent(xaif, speaker=speaker))
        spkr_analytic_list.append(analytics.ra_in_divergent(xaif, speaker=speaker))
        spkr_analytic_list.append(analytics.ra_in_linked(xaif, speaker=speaker))

        spkr_analytic_list.append(analytics.premise_count(xaif, speaker=speaker))
        spkr_analytic_list.append(analytics.concl_count(xaif, speaker=speaker))
        spkr_analytic_list.append(analytics.prem_concl_ratio(xaif, speaker=speaker))
        spkr_analytic_list.append(analytics.ra_ca_ratio(xaif, speaker=speaker))

    
        if spkr_analytic_list != []:
            for s in spkr_rel_counts.keys():
                # rel_counts[s].update(concl_first[s])
                for a in spkr_analytic_list:
                    spkr_rel_counts[s].update(a[s])
                

    # Global analytics
    global_analytic_list = []
    global_analytic_list.append(analytics.map_wordcount(xaif))
    global_analytic_list.append(analytics.loc_counts(xaif, speaker=False))
    global_analytic_list.append(analytics.arg_word_densities(xaif, speaker=False))
    global_analytic_list.append(analytics.arg_loc_densities(xaif, speaker=False))

    global_analytic_list.append(analytics.ra_in_serial(xaif, speaker=False))
    global_analytic_list.append(analytics.ra_in_convergent(xaif, speaker=False))
    global_analytic_list.append(analytics.ra_in_divergent(xaif, speaker=False))
    global_analytic_list.append(analytics.ra_in_linked(xaif, speaker=False))

    global_analytic_list.append(analytics.premise_count(xaif, speaker=False))
    global_analytic_list.append(analytics.concl_count(xaif, speaker=False))
    global_analytic_list.append(analytics.prem_concl_ratio(xaif, speaker=False))
    global_analytic_list.append(analytics.ra_ca_ratio(xaif, speaker=False))

    #Adding analytics which calculate 'per node'
    if node_level:
        node_analytic_list = []
        node_analytic_list.append(analytics.node_wc(xaif))
        node_analytic_list.append(analytics.supportedNodes(xaif))
        node_analytic_list.append(analytics.attackedNodes(xaif))
        node_analytic_list.append(analytics.nodeTenseScores(xaif))
        node_analytic_list.append(analytics.ner(xaif))
        node_analytic_list.append(analytics.sentiment(xaif))
        print(node_analytic_list)

        # print(node_analytic_list)


        #Forecast-specific analytics
    if forecast:
        forecast_analytics_list = []
        forecast_analytics_list.append(analytics.addForecastAccuracy(xaif))
        forecast_analytics_list.append(analytics.addNodeOutcomes(xaif))
        subgraphs = analytics.getHypSubgraphs(xaif)
        forecast_analytics_list.append(analytics.raCount(xaif))

        for graph in subgraphs:
            forecast_analytics_list.append(analytics.raCount(graph))
    
    xaif['analytics'] = {}
    xaif['analytics']['global'] = global_analytic_list

    if speaker:
        xaif['analytics'] = {"speaker": spkr_rel_counts}
    
    if node_level:
        xaif['analytics']['node'] = node_analytic_list
    if forecast:
        xaif['analytics']['forecast'] = forecast_analytics_list
    
    # print(xaif)
    return xaif
