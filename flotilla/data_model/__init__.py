"""
data model
"""

from .expression import ExpressionData
from .gene_ontology import GeneOntologyData
from .metadata import MetaData
from .quality_control import MappingStatsData
from .splicing import SplicingData

__author__ = 'olga'

__all__ = ['ExpressionData', 'GeneOntologyData',
           'MetaData', 'MappingStatsData', 'SplicingData']
