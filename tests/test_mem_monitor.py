import csv
import tempfile
import unittest
from unittest import mock

from utils.mem_monitor import MemMonitor


class MemMonitorTest(unittest.TestCase):
    def test_uss_sampled_only_every_n_ticks_and_csv_layout_unchanged(self):
        with tempfile.TemporaryDirectory() as d:
            mon = MemMonitor(log_dir=d, warn_avail_gb=0.0, flush_every=1, uss_every=3)
            with mock.patch.object(mon, "_sample_uss_gb", wraps=mon._sample_uss_gb) as sampler:
                for it in range(1, 8):
                    mon.tick(it)
            mon.close()
            self.assertEqual(sampler.call_count, 3)  # ticks 0, 3, 6 of 7
            rows = list(csv.reader(open(f"{d}/mem_monitor.csv")))
            self.assertEqual(rows[0], ["iter", "wall_s", "rss_gb", "uss_gb", "tree_rss_gb",
                                       "avail_gb", "total_gb", "swap_used_gb", "num_children"])
            self.assertEqual(len(rows), 8)
            self.assertTrue(all(len(r) == 9 for r in rows[1:]))
            self.assertEqual([r[0] for r in rows[1:]], [str(i) for i in range(1, 8)])
            uss = [r[3] for r in rows[1:]]
            self.assertNotEqual(uss[0], "N/A")            # sampled on the first tick
            self.assertEqual(uss[1], uss[0])              # repeated in between
            self.assertTrue(float(rows[1][2]) > 0)        # rss still per tick

    def test_uss_can_be_disabled(self):
        with tempfile.TemporaryDirectory() as d:
            mon = MemMonitor(log_dir=d, uss_every=0)
            with mock.patch.object(mon, "_sample_uss_gb") as sampler:
                mon.tick(1); mon.tick(2)
            mon.close()
            sampler.assert_not_called()
            rows = list(csv.reader(open(f"{d}/mem_monitor.csv")))
            self.assertEqual([r[3] for r in rows[1:]], ["N/A", "N/A"])


if __name__ == "__main__":
    unittest.main()
