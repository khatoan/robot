#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
import os
from glob import glob

package_name = "robot"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
        (os.path.join("share", package_name, "config"), glob("config/*")),
    ],
    install_requires=["setuptools"],
    python_requires=">=3.10",
    zip_safe=True,
    maintainer="lktoan",
    maintainer_email="lktoan@example.com",
    description="Rescue robot stack: perception, fusion, decision and communication nodes",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "perception_node = robot.perception_node:main",
            "fusion_node = robot.fusion_node:main",
            "decision_node = robot.decision_node:main",
            "communication_node = robot.communication_node:main",
        ],
    },
)
