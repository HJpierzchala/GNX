# STEC network of 2024-035

Input data of the tutorials `activity_indexes.ipynb` and `kriging.ipynb`: slant TEC of 38 EUREF Permanent
Network (EPN) stations on 2024-02-04 (day 035), one file per station and system, `<STATION>_<G|E>_STEC.parquet.gzip`
(75 files: LAMA has no Galileo file, because its receiver flags almost every Galileo E5a phase observation with a
loss of lock, so no Galileo arc survives the cycle-slip detection).

Each file is the output of one `gnx_py.ionosphere.TECSession` run of GNX-py v1, indexed by `(sv, time)` (arc label,
naive GPS time), with STEC in el/m^2 (`leveled_tec`, `median_leveled_tec`, `code_tec`), the elevation and azimuth,
the ionospheric pierce point (`lat_ipp`, `lon_ipp`), the GIM slant delay `ion` and the bias diagnostics
(`sta_dcb`, `stec_station_bias_source`, `stec_sat_bias_source`, `stec_station_bias_m`, `stec_sat_bias_m`).

Processing settings:

- observations: EPN daily RINEX files, 30 s, from the BKG GNSS Data Center;
- signals: GPS L1/L2 (`gps_freq="L1L2"`), Galileo E1/E5a (`gal_freq="E1E5a"`); elevation mask 30 deg;
- orbits: CODE final orbits (`COD0OPSFIN` SP3 of days 034-036, `orbit_type="precise"`) and the IGS broadcast
  navigation `BRDC00IGS_R_20240350000_01D_MN.rnx` (`nav_path`); satellite antenna offsets along the line of sight
  (`sat_pco="los"`) and receiver antenna offsets from the IGS20 ANTEX of GPS week 2388 (`igs20_2388.atx`); no
  phase wind-up and no relativistic path term; `screen=True`, G02 and G20 left out (`skip_sat`), as in the
  original processing of this network, whose settings these files reproduce;
- ionosphere: the CODE GIM `COD0OPSFIN_20240350000_01D_01H_GIM.INX` (`ionosphere_model="gim"`), whose header
  also provides the satellite DCBs (no `dcb_path`);
- receiver DCB: calibrated against the CODE GIM (`add_sta_dcb=True`, `rcv_dcb_source="calibrate"`), so the
  absolute level of the network is tied to that GIM;
- levelling windows of 50 samples (`leveling_ws=50`, `median_leveling_ws=50`), minimum arc length 2. The
  `TECConfig` default levels the whole arc (`leveling_ws=None`, as in `ionosphere_introduction.ipynb`); the
  windows are kept here to reproduce the original processing.

The files are GNX results; the observations and products they come from remain the work of their producers
(see `THIRD_PARTY_NOTICES`). The EPN observations of all 38 stations are distributed under CC BY 4.0 (EPN Data
Access and Use Policy, https://epncb.eu/_documentation/EPNDataAccessandUsePolicy_2026.pdf): when you use or
redistribute these files, credit the EPN and its station operators and say that the observations were processed
into slant TEC with GNX-py.
