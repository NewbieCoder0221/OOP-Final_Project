"""
report_generator.py
Generates data quality report and health score.
"""

class ReportGenerator:
    """
    Produces reports and evaluates data quality.
    """

    def __init__(self, doctor):
        """
        doctor: instance of DataDoctor
        """
        self._doctor = doctor
        self._report_data = {}
        self._score = 0

    def report(self):
        """
        Generate a human-readable summary of issues and fixes.
        """
        self._report_data = {
            "issues": self._doctor.get_summary(),
            "fixes": self._doctor.get_fix_log()
        }
        return self._report_data

    def health_score(self):
        """
        Calculates a simple data score (Week 2 basic version).
        """
        issues = self._doctor.get_summary()
        
        missing = issues["missing"].sum()
        duplicates = issues["duplicates"]
        outliers = sum(issues["outliers"].values())

        total_penalty = missing + duplicates + outliers
        score = max(0, 100 - total_penalty)
        self._score = score

        return score

    def export_report(self, file_name="report.txt"):
        """
        Exports the report to a text file.
        """
        with open(file_name, "w") as f:
            f.write(str(self._report_data))
