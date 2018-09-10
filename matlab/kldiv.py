#!/usr/bin/env python
#
# Copyright 2014 Quantopian, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def kld(target, rho):
    """ calculate kl divergence (relationship) between 2 distributions
    """

    kl = []
    for val in rho:
        kl.append(target*np.log(target/val) + (1-target)*np.log((1-target)/(1-val)))
    return kl

# Note: this function can be removed if running
# this algorithm on quantopian.com
def main():
    rho = np.arange(0.01, 1, 0.01)  

    plt.figure(1, figsize = (18,8))
    plt.suptitle('kl divergence')
    plt.xlim((0,1))
    plt.ylim((0,6))
    plt.grid(True)

    ax1 = plt.subplot(111)
    ax1.plot(rho, kld(0.2, rho))
    ax1.set_xlabel('rho')
    ax1.set_ylabel('KL Divergence')

    #ax2 = plt.subplot(212, sharex=ax1)
    #results.AAPL.plot(ax=ax2)
    #ax2.set_ylabel('AAPL price (USD)')

    # Show the plot.
    plt.show()


if __name__ == "__main__":
    main()

