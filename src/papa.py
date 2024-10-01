import json
from . import xaif_toolbox as ova3
from . import analytics
import sys
import os

def all_analytics(xaif, node_level=False, skipDialog=False, forecast=False):
    # This involves redundancy currently bc all analytics are designed to call the info collection
    # so that they can be run individually with just the XAIF...


    # Speaker-level analytics
    analytic_list = []
    rel_counts = analytics.arg_relation_counts(xaif)
    # print(rel_counts)

    xaif = ova3.ova2_to_ova3(xaif)
    #For now skipping thing that don't work/aren't needed for forecast
    if not skipDialog:
        if 'AIF' in xaif.keys():
            wordcounts = ova3.spkr_wordcounts(xaif)
        analytic_list.append(wordcounts)
        analytic_list.append(analytics.concl_first_perc(xaif))
        analytic_list.append(analytics.arg_word_densities(xaif))
        analytic_list.append(analytics.arg_loc_densities(xaif))
    analytic_list.append(analytics.ra_in_serial(xaif))
    analytic_list.append(analytics.ra_in_convergent(xaif))
    analytic_list.append(analytics.ra_in_divergent(xaif))
    analytic_list.append(analytics.ra_in_linked(xaif))
    analytic_list.append(analytics.avg_arg_depths(xaif))
    analytic_list.append(analytics.avg_arg_breadths(xaif))
    analytic_list.append(analytics.arg_intros(xaif))
    analytic_list.append(analytics.direct_args_from_others(xaif))
    analytic_list.append(analytics.indirect_args_from_others(xaif))

    # concl_first = analytics.concl_first_perc(xaif)
    # if not skipDialog:
    #     arg_densities = analytics.arg_densities(xaif)

    for s in rel_counts.keys():
        # rel_counts[s].update(concl_first[s])
        for a in analytic_list:
            rel_counts[s].update(a[s])

    #Forecast-specific analytics
    if forecast:
        forecast_analytics_list = []
        forecast_analytics_list.append(analytics.addForecastAccuracy(xaif))
        forecast_analytics_list.append(analytics.addNodeOutcomes(xaif))

    #Adding analytics which calculate 'per node'
    if node_level:
        node_analytic_list = []
        node_analytic_list.append(analytics.node_wc(xaif))
        node_analytic_list.append(analytics.supportedNodes(xaif))
        node_analytic_list.append(analytics.attackedNodes(xaif))
        node_analytic_list.append(analytics.nodeTenseScores(xaif))
        node_analytic_list.append(analytics.ner(xaif))
        # print(node_analytic_list)
        # xaif['analytics']['node'] = node_analytic_list
    
    
    xaif['analytics'] = {
        "speaker": rel_counts
    }
    if node_level:
        xaif['analytics']['node'] = node_analytic_list
    if forecast:
        xaif['analytics']['forecast'] = forecast_analytics_list
    
    # print(xaif)
    return xaif
