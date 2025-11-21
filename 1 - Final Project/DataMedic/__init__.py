from .data_inspector import DataInspector
from .data_doctor import DataDoctor
from .report_generator import ReportGenerator
from .data_getter import DataGetter  # Added import for DataGetter

__all__ = ["DataInspector", "DataDoctor", "ReportGenerator", "DataGetter"]  # Added DataGetter to __all__


__version__ = "0.1.0"
__author__ = "Your Group Name"
