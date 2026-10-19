"""Settlement stations for every Polymarket daily-temperature city.

Each Polymarket "Highest/Lowest temperature in <city> on <date>?" event names
one observing station in its rules (almost always an airport METAR site shown
on weather.gov/wrh/timeseries). The forecast has to be for *that* point, not
the city centre: LaGuardia vs Central Park or London City vs Heathrow differ
by a degree or more on many days.

`city` keys are the slug fragment Polymarket uses (`highest-temperature-in-<city>-on-...`).
Coordinates/elevations come from aviationweather.gov station info.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Station:
    city: str
    icao: str
    name: str
    lat: float
    lon: float
    elevation_m: float
    tz: str
    unit: str  # "F" or "C": the unit the market's buckets are written in
    # "metar": settles on METAR obs (weather.gov timeseries / Wunderground)
    # "hko": Hong Kong Observatory daily extract (not a METAR site)
    source: str = "metar"
    nbm: bool = False  # US station covered by NOAA's National Blend of Models text product

    @property
    def is_us(self) -> bool:
        return self.icao.startswith("K")


def _s(city, icao, name, lat, lon, elev, tz, unit, source="metar"):
    return Station(city, icao, name, lat, lon, elev, tz, unit, source, nbm=icao.startswith("K"))


STATIONS = {s.city: s for s in [
    # --- United States (whole °F, NBM available) ---
    _s("nyc", "KLGA", "LaGuardia Airport", 40.77945, -73.88027, 9, "America/New_York", "F"),
    _s("atlanta", "KATL", "Hartsfield-Jackson Intl", 33.62972, -84.44223, 309, "America/New_York", "F"),
    _s("miami", "KMIA", "Miami Intl", 25.78806, -80.31692, 1, "America/New_York", "F"),
    _s("chicago", "KORD", "Chicago O'Hare Intl", 41.96017, -87.93161, 202, "America/Chicago", "F"),
    _s("dallas", "KDAL", "Dallas Love Field", 32.83836, -96.83584, 148, "America/Chicago", "F"),
    _s("austin", "KAUS", "Austin-Bergstrom Intl", 30.1831, -97.68063, 148, "America/Chicago", "F"),
    _s("houston", "KHOU", "William P. Hobby", 29.64582, -95.28214, 13, "America/Chicago", "F"),
    _s("denver", "KBKF", "Buckley Space Force Base", 39.713, -104.758, 1703, "America/Denver", "F"),
    _s("seattle", "KSEA", "Seattle-Tacoma Intl", 47.44467, -122.31442, 115, "America/Los_Angeles", "F"),
    _s("los-angeles", "KLAX", "Los Angeles Intl", 33.93817, -118.3866, 30, "America/Los_Angeles", "F"),
    _s("san-francisco", "KSFO", "San Francisco Intl", 37.61961, -122.36561, 2, "America/Los_Angeles", "F"),
    # --- Americas (°C) ---
    _s("toronto", "CYYZ", "Toronto Pearson Intl", 43.679, -79.629, 171, "America/Toronto", "C"),
    _s("mexico-city", "MMMX", "Benito Juarez Intl", 19.436, -99.072, 2224, "America/Mexico_City", "C"),
    _s("panama-city", "MPMG", "Marcos A. Gelabert Intl", 8.967, -79.555, 6, "America/Panama", "C"),
    _s("sao-paulo", "SBGR", "Sao Paulo-Guarulhos Intl", -23.432, -46.469, 745, "America/Sao_Paulo", "C"),
    _s("buenos-aires", "SAEZ", "Minister Pistarini Intl", -34.822, -58.536, 16, "America/Argentina/Buenos_Aires", "C"),
    # --- Europe ---
    _s("london", "EGLC", "London City Airport", 51.505, 0.055, 10, "Europe/London", "C"),
    _s("paris", "LFPB", "Paris-Le Bourget", 48.967, 2.428, 50, "Europe/Paris", "C"),
    _s("amsterdam", "EHAM", "Amsterdam Schiphol", 52.315, 4.79, -2, "Europe/Amsterdam", "C"),
    _s("munich", "EDDM", "Munich Airport", 48.348, 11.813, 445, "Europe/Berlin", "C"),
    _s("milan", "LIMC", "Malpensa Intl", 45.631, 8.728, 221, "Europe/Rome", "C"),
    _s("madrid", "LEMD", "Madrid-Barajas", 40.466, -3.555, 589, "Europe/Madrid", "C"),
    _s("warsaw", "EPWA", "Warsaw Chopin", 52.163, 20.961, 107, "Europe/Warsaw", "C"),
    _s("helsinki", "EFHK", "Helsinki Vantaa", 60.327, 24.957, 56, "Europe/Helsinki", "C"),
    _s("moscow", "UUWW", "Vnukovo Intl", 55.592, 37.261, 195, "Europe/Moscow", "C"),
    _s("istanbul", "LTFM", "Istanbul Airport", 41.262, 28.74, 99, "Europe/Istanbul", "C"),
    _s("ankara", "LTAC", "Esenboga Intl", 40.128, 32.995, 952, "Europe/Istanbul", "C"),
    # --- Middle East / Africa / South Asia ---
    _s("tel-aviv", "LLBG", "Ben Gurion Intl", 32.011, 34.887, 35, "Asia/Jerusalem", "C"),
    _s("jeddah", "OEJN", "King Abdulaziz Intl", 21.685, 39.166, 8, "Asia/Riyadh", "C"),
    _s("karachi", "OPKC", "Jinnah Intl", 24.902, 67.139, 20, "Asia/Karachi", "C"),
    _s("lucknow", "VILK", "Chaudhary Charan Singh Intl", 26.761, 80.889, 121, "Asia/Kolkata", "C"),
    _s("cape-town", "FACT", "Cape Town Intl", -33.965, 18.602, 48, "Africa/Johannesburg", "C"),
    # --- East / Southeast Asia, Oceania ---
    _s("singapore", "WSSS", "Changi Airport", 1.368, 103.982, 17, "Asia/Singapore", "C"),
    _s("kuala-lumpur", "WMKK", "Kuala Lumpur Intl", 2.747, 101.714, 21, "Asia/Kuala_Lumpur", "C"),
    _s("manila", "RPLL", "Ninoy Aquino Intl", 14.507, 121.004, 15, "Asia/Manila", "C"),
    _s("hong-kong", "VHHH", "Hong Kong Observatory", 22.302, 114.174, 32, "Asia/Hong_Kong", "C", source="hko"),
    _s("shenzhen", "ZGSZ", "Shenzhen Bao'an Intl", 22.639, 113.803, 18, "Asia/Shanghai", "C"),
    _s("guangzhou", "ZGGG", "Guangzhou Baiyun Intl", 23.392, 113.307, 11, "Asia/Shanghai", "C"),
    _s("shanghai", "ZSPD", "Shanghai Pudong Intl", 31.146, 121.8, 4, "Asia/Shanghai", "C"),
    _s("beijing", "ZBAA", "Beijing Capital Intl", 40.082, 116.603, 31, "Asia/Shanghai", "C"),
    _s("chongqing", "ZUCK", "Chongqing Jiangbei Intl", 29.718, 106.639, 416, "Asia/Shanghai", "C"),
    _s("chengdu", "ZUUU", "Chengdu Shuangliu Intl", 30.576, 103.95, 494, "Asia/Shanghai", "C"),
    _s("wuhan", "ZHHH", "Wuhan Tianhe Intl", 30.783, 114.205, 33, "Asia/Shanghai", "C"),
    _s("qingdao", "ZSQD", "Qingdao Jiaodong Intl", 36.362, 120.087, 2, "Asia/Shanghai", "C"),
    _s("jinan", "ZSJN", "Jinan Yaoqiang Intl", 36.856, 117.206, 22, "Asia/Shanghai", "C"),
    _s("zhengzhou", "ZHCC", "Zhengzhou Xinzheng Intl", 34.52, 113.834, 149, "Asia/Shanghai", "C"),
    _s("taipei", "RCSS", "Taipei Songshan", 25.069, 121.552, 8, "Asia/Taipei", "C"),
    _s("tokyo", "RJTT", "Tokyo Haneda", 35.553, 139.781, 5, "Asia/Tokyo", "C"),
    _s("seoul", "RKSI", "Incheon Intl", 37.469, 126.451, 7, "Asia/Seoul", "C"),
    _s("busan", "RKPK", "Gimhae Intl", 35.179, 128.938, 3, "Asia/Seoul", "C"),
    _s("wellington", "NZWN", "Wellington Intl", -41.331, 174.806, 12, "Pacific/Auckland", "C"),
]}

_BY_ICAO = {s.icao: s for s in STATIONS.values()}


def station_for(city: str, icao: Optional[str] = None) -> Optional[Station]:
    """Look up by city slug, falling back to the ICAO code parsed from the market rules.

    The ICAO in the rules wins if it disagrees with our table: Polymarket has
    moved cities between stations before (e.g. NYC once settled on Wunderground KLGA,
    now NOAA KLGA), and a silent station change would poison every forecast.
    """
    s = STATIONS.get(city)
    if icao:
        icao = icao.upper()
        if s is not None and s.icao != icao and s.source == "metar":
            return _BY_ICAO.get(icao)
        if s is None:
            return _BY_ICAO.get(icao)
    return s
