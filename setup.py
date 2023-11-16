from setuptools import setup
import versioneer

setup(
    name="mordor",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author="Jonas Witthuhn",
    author_email="witthuhn@tropos.de",
    license="OSI Approved :: GNU General Public License v3 (GPLv3)",
    packages=["mordor"],
    package_dir={"": "src"},
    package_data={"mordor": ["share/*.json", "share/*.CR1X"]},
    include_package_data=True,
    scripts= [
            "bin/wiser_move.bat",
            "bin/asi16_move.bat"
        ],
    entry_points={
        "console_scripts": [
            "mordor = mordor.click:cli",
            "cloud_coverage = mordor:cloud_coverage",
        ],
    },
    install_requires=[
        "numpy",
        "xarray",
        "netcdf4",
        "pip",
        "jstyleson",
        "Click",
        "toolz",
        "addict",
        "trosat-base @ git+https://github.com/hdeneke/trosat-base.git#egg=trosat-base",
        "argparse",
        "configparser",
        "pandas",
        "psycopg2",
    ],
    extras_require={
        "docs": [
            "sphinx",
            "myst-parser",
            "myst-nb",
        ],
        "nbs": [
            "jupyter",
            "nbdev",
            "nbformat",
        ],
    },
)
