# pylint: disable=E1101,E0611,W0231,W0201
# E1101 throws errors on my setattr() stuff and astropy.units.W and astropy.units.Hz
# E0611 throws an error when importing astropy.cosmology.Planck15
# W0231 gives a warning because __init__() is not called for BaseValidationTest
# W0201 gives a warning when defining attributes outside of __init__()
from __future__ import unicode_literals, absolute_import, division
import os
import numpy as np
from astropy import units as u
from astropy import constants as c
from astropy.cosmology import Planck15 as cosmo
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from GCR import GCRQuery
from pandas import read_csv
from descqa import BaseValidationTest, TestResult

emline_names = {'ha': r'H$\alpha$', 'hb': r'H$\beta$', 'oii': '[OII]', 'oiii': '[OIII]'}

__all__ = ['EmlineRatioTest']

class EmlineEWTest(BaseValidationTest):
    """
    Validation test for H-alpha luminosity vs. equivalent width to validate emission line normalization relative to continuum.

    Parameters
    ----------
    sdss_file: str, optional, (default: 'sdss_emission_lines/sdss_query_snr10_ew.csv')
        Location of the SDSS data file that will be passed into the sdsscat class.  Looks
        in the 'data/' folder.
    mag_u_cut: float, optional, (default: 26.3)
        u-band magnitude cut.  Dimmer galaxies are excluded from the comparison.  Default
        value is the 5-sigma detection limit from https://arxiv.org/pdf/0912.0201.pdf
    mag_g_cut: float, optional, (default: 27.5)
        g-band magnitude cut.  Dimmer galaxies are excluded from the comparison.  Default
        value is the 5-sigma detection limit from https://arxiv.org/pdf/0912.0201.pdf
    mag_r_cut: float, optional, (default: 27.7)
        r-band magnitude cut.  Dimmer galaxies are excluded from the comparison.  Default
        value is the 5-sigma detection limit from https://arxiv.org/pdf/0912.0201.pdf
    mag_i_cut: float, optional, (default: 27.0)
        i-band magnitude cut.  Dimmer galaxies are excluded from the comparison.  Default
        value is the 5-sigma detection limit from https://arxiv.org/pdf/0912.0201.pdf
    mag_z_cut: float, optional, (default: 26.2)
        z-band magnitude cut.  Dimmer galaxies are excluded from the comparison.  Default
        value is the 5-sigma detection limit from https://arxiv.org/pdf/0912.0201.pdf
    mag_y_cut: float, optional, (default: 24.9)
        y-band magnitude cut.  Dimmer galaxies are excluded from the comparison.  Default
        value is the 5-sigma detection limit from https://arxiv.org/pdf/0912.0201.pdf
    sdss_drawnum: int, optional, (default: 30000)
        The number of galaxies to draw from the SDSS data file to perform the comparison.
        The default number is chosen to (hopefully) not make the 2-D KS test too stringent.
    sim_drawnum: int, optional, (default: 30000)
        The number of galaxies to draw from the simulated data to perform the comparison.
        The default number is chosen to (hopefully) not make the 2-D KS test too stringent.
    truncate_cat_name: Bool, optional, (default: False)
        Specifies whether the catalog name displayed in the summary figure should be 
        shortened.
    """
    def __init__(self, **kwargs):

        np.random.seed(0)

        # load test config options
        self.kwargs = kwargs
        sdss_file = kwargs.get('sdss_file', 'descqa/data/sdss_emission_lines/sdss_query_snr10_ew.csv')
        # self.sdsscat = sdsscat(self.data_dir + '/' + sdss_file)
        self.sdsscat = sdsscat(sdss_file)

        # The magnitude cuts for galaxies pulled from the catalog.  These numbers correspond to
        # a 5-sigma cut based on https://arxiv.org/pdf/0912.0201.pdf

        self.mag_u_cut = kwargs.get('mag_u_cut', 26.3)
        self.mag_g_cut = kwargs.get('mag_g_cut', 27.5)
        self.mag_r_cut = kwargs.get('mag_r_cut', 27.7)
        self.mag_i_cut = kwargs.get('mag_i_cut', 27.0)
        self.mag_z_cut = kwargs.get('mag_z_cut', 26.2)
        self.mag_y_cut = kwargs.get('mag_y_cut', 24.9)

        # If None, these will use the full sample.  Otherwise, it will draw a 
        # random sample with the given size

        self.sdss_drawnum = kwargs.get('sdss_drawnum', None)
        self.sim_drawnum = kwargs.get('sim_drawnum', None)

        self.figlist = []
        self.runcat_name = []

        self.truncate_cat_name = kwargs.get('truncate_cat_name', False)


    def run_on_single_catalog(self, catalog_instance, catalog_name, output_dir):

        #=========================================
        # Begin Reading in Data
        #=========================================

        # check if needed quantities exist
        if not catalog_instance.has_quantities(['mag_u_lsst',
                                                'mag_g_lsst',
                                                'mag_r_lsst',
                                                'mag_i_lsst',
                                                'mag_z_lsst',
                                                'mag_y_lsst',
                                                'emissionLines/totalLineLuminosity:balmerAlpha6563',
                                                'sed_6548_406']):
            return TestResult(skipped=True, summary='Necessary quantities are not present')

        uband_maglim = GCRQuery((np.isfinite, 'mag_u_lsst'), 'mag_u_lsst < %.1f' % self.mag_u_cut)
        gband_maglim = GCRQuery((np.isfinite, 'mag_g_lsst'), 'mag_g_lsst < %.1f' % self.mag_g_cut)
        rband_maglim = GCRQuery((np.isfinite, 'mag_r_lsst'), 'mag_r_lsst < %.1f' % self.mag_r_cut)
        iband_maglim = GCRQuery((np.isfinite, 'mag_i_lsst'), 'mag_i_lsst < %.1f' % self.mag_i_cut)
        zband_maglim = GCRQuery((np.isfinite, 'mag_z_lsst'), 'mag_z_lsst < %.1f' % self.mag_z_cut)
        yband_maglim = GCRQuery((np.isfinite, 'mag_y_lsst'), 'mag_y_lsst < %.1f' % self.mag_y_cut)


        data = catalog_instance.get_quantities(['emissionLines/totalLineLuminosity:balmerAlpha6563',
                                                'sed_6548_406',
                                                'mag_u_lsst',
                                                'mag_g_lsst',
                                                'mag_r_lsst',
                                                'mag_i_lsst',
                                                'mag_z_lsst',
                                                'mag_y_lsst'], filters=(uband_maglim | gband_maglim | rband_maglim | iband_maglim | zband_maglim | yband_maglim))
        ha_lum = data['emissionLines/totalLineLuminosity:balmerAlpha6563'] * 3.839e26*u.W
        lnu_continuum = data['sed_6548_406'] * 4.4659e13*u.W/u.Hz

        # Reduce the sample size by drawing self.sim_drawnum galaxies

        if self.sim_drawnum != None:
            indices = np.random.choice(np.arange(len(ha_lum)), size=self.sim_drawnum, replace=False)
            ha_lum = ha_lum[indices]
            lnu_continuum = lnu_continuum[indices]

        llam_continuum = lnu_continuum * c.c / ((6548 + 0.5*406)*u.Angstrom)**2

        self.ha_lum = ha_lum.to('erg/s').value
        self.ha_ew = (ha_lum / llam_continuum).to('Angstrom').value


        #=========================================
        # End Reading in Data
        #=========================================

        #=========================================
        # Perform the Test and Return Results
        #=========================================

        if self.truncate_cat_name:
            self.makeplot(catalog_name.split('_')[0])
        else:
            self.makeplot(catalog_name)

        self.figlist.append(thisfig)
        self.runcat_name.append(catalog_name)

        return TestResult(inspect_only=True, summary='Eventually, let\'s modify this so that there isa  numerical test!')



    def makeplot(self, catalog_name, x_range=(-4, 4), y_range=(35, 45)):
        """
        Make a summary plot of the test results
        """
        #=========================================
        # Begin Test and Plotting
        #=========================================

        fig = plt.figure(figsize=(16, 8))
        sp1 = fig.add_subplot(121)
        sp2 = fig.add_subplot(122)

        sdss_dist = np.vstack((np.log10(self.sdsscat.ha_ew), np.log10(self.sdsscat.ha_lum)))
        sim_dist = np.vstack((np.log10(self.ha_ew), np.log10(self.ha_lum)))

        xlabel = ''
        ylabel = ''

        # Generate each distribution

        sp1.hist2d(*sdss_dist, bins=50, range=[x_range, y_range], norm=LogNorm(), cmap='plasma_r')
        sp2.hist2d(*sim_dist, bins=50, range=[x_range, y_range], norm=LogNorm(), cmap='plasma_r')

        # Draw a number of SDSS galaxies equal to self.sdss_drawnum

        if self.sdss_drawnum != None:
            sdss_draw_inds = np.random.choice(np.arange(len(sdss_dist[0])), size=self.sdss_drawnum)
            sdss_dist = sdss_dist[:, sdss_draw_inds]

        # Plotting stuff

        xlabel = r'$\log_{10}$(EW$_\mathrm{H\alpha}$)'
        ylabel = r'$\log_{10}$(L$_\mathrm{H\alpha}$)'

        sp1.set_xlabel(xlabel, fontsize=20)
        sp1.set_ylabel(ylabel, fontsize=20)
        sp2.set_xlabel(xlabel, fontsize=20)
        sp1.set_xlim(x_range)
        sp1.set_ylim(y_range)
        sp2.set_xlim(x_range)
        sp2.set_ylim(y_range)

        sp2.set_yticklabels([])

        plt.subplots_adjust(wspace=0.0)

        sp1.text(0.98, 0.02, 'SDSS', fontsize=24, ha='right', va='bottom', transform=sp1.transAxes)
        sp2.text(0.98, 0.02, catalog_name, fontsize=24, ha='right', va='bottom', transform=sp2.transAxes)



    def summary_file(self, output_dir):
        """
        Saves a summary file with information about the cuts performed on the data in order to
        perform the test
        """

        with open(os.path.join(output_dir, 'Emline_Lum_Ratio_Summary.txt'), 'w') as writefile:
            if self.sim_drawnum != None:
                writefile.write('Simulation Galaxies Drawn: %i\n' % self.sim_drawnum)
            else:
                writefile.write('Used all simulation galxies.')

            if self.sdss_drawnum != None:
                writefile.write('SDSS Galaxies Drawn: %i\n' % self.sdss_drawnum)
            else:
                writefile.write('Used all SDSS galaxies.')

            for thisband in ['u', 'g', 'r', 'i', 'z', 'y']:
                writefile.write(thisband + '-band magnitude cut: %.1f\n' % getattr(self, 'mag_' + thisband + '_cut'))
            writefile.write('\n')
            writefile.write('=================\n')
            writefile.write(' Catalogs Tested \n')
            writefile.write('=================\n')


            for thiscat in self.runcat_name:

                writefile.write(thiscat + '\n')






    def conclude_test(self, output_dir):

        # Save a summary file with the details of the test

        self.summary_file(output_dir)

        # Save all of the summary plots into output_dir

        for thisfig, thiscat in zip(self.figlist, self.runcat_name):
            thisfig.savefig(os.path.join(output_dir, thiscat + '_emline_equiv_widths.png'), bbox_inches='tight')
            plt.close(thisfig)





class sdsscat:
    """
    This class holds the SDSS data in an easily accessible form, and also dust corrects
    the emission lines using the Balmer Decrement.
    """


    def __init__(self, infile):

        self.Calzetti2000 = np.vectorize(self.Calzetti2000_novec)

        data = read_csv(infile)

        usecols = ['z', 'z_err', 'oii_flux', 'oii_flux_err', 'oiii_flux', 'oiii_flux_err',
                   'h_alpha_flux', 'h_alpha_flux_err', 'h_beta_flux', 'h_beta_flux_err',
                   'lgm_tot_p50', 'lgm_tot_p16', 'lgm_tot_p84', 'sfr_tot_p50', 'sfr_tot_p16', 'sfr_tot_p84',
                   'oh_p50', 'h_alpha_eqw', 'oiii_4959_eqw', 'oiii_5007_eqw', 'oii_3726_eqw', 'oii_3729_eqw', 'h_beta_eqw',
                   'h_alpha_eqw_err', 'oiii_4959_eqw_err', 'oiii_5007_eqw_err', 'oii_3726_eqw_err', 'oii_3729_eqw_err', 'h_beta_eqw_err']
        newnames = ['z', 'z_err', 'oii_uncorr', 'oii_err_uncorr', 'oiii_uncorr', 'oiii_err_uncorr', 'ha_uncorr', 'ha_err_uncorr', 'hb_uncorr', 'hb_err_uncorr',
                    'logmstar', 'logmstar_lo', 'logmstar_hi', 'sfr', 'sfr_lo', 'sfr_hi', 'o_abundance', 'ha_ew_uncorr', 'oiii4959_ew_uncorr', 'oiii5007_ew_uncorr', 'oii3726_ew_uncorr', 'oii3729_ew_uncorr', 'hb_ew_uncorr',
                    'ha_ew_err', 'oiii4959_ew_err_uncorr', 'oiii5007_ew_err_uncorr', 'oii3726_ew_err_uncorr', 'oii3729_ew_err_uncorr', 'hb_ew_err_uncorr']

        for col, name in zip(usecols, newnames):
            setattr(self, name, data[col].values)

        for x, colname in enumerate(newnames):
            if 'flux' in usecols[x]:
                setattr(self, colname, getattr(self, colname)/10**17) # Units are 10**-17 erg/s/cm^2

        # Dust correction
        # E(B-V) = log_{10}(ha_uncorr/(hb_uncorr*2.86)) *(-0.44/0.4) / (k(lam_ha) - k(lam_hb))

        self.EBV = np.log10(self.ha_uncorr/(self.hb_uncorr*2.86)) * (-.44/0.4) / (self.Calzetti2000(6563.) - self.Calzetti2000(4863.))

        # A_oiii = self.Calzetti2000(4980.) * self.EBV / 0.44
        # A_oii = self.Calzetti2000(3727.) * self.EBV / 0.44
        # A_ha = self.Calzetti2000(6563.) * self.EBV / 0.44
        # A_hb = self.Calzetti2000(4863.) * self.EBV / 0.44

        for x, colname in enumerate(newnames):

            if 'ha_' in colname:
                wave = 6563.
            elif 'hb_' in colname:
                wave = 4863.
            elif 'oii_' in colname:
                wave = 3727.
            elif 'oiii_' in colname:
                wave = 4980.
            elif 'oii3726_' in colname:
                wave = 3726.
            elif 'oii3729_' in colname:
                wave = 3729.
            elif 'oiii4959_' in colname:
                wave = 4969.
            elif 'oiii5007_' in colname:
                wave = 5007.

            if 'uncorr' in colname and 'ew' not in colname:

                A_line = self.Calzetti2000(wave) * self.EBV / 0.44

                newflux = getattr(self, colname) * np.power(10, 0.4*A_line)
                setattr(self, colname[:-7], newflux)

            elif 'uncorr' in colname and 'ew' in colname:

                multiplier = np.power(10, 0.4 * self.Calzetti2000(wave) * self.EBV * ((1./.44) - 1.))
                setattr(self, colname[:-7], getattr(self, colname)*multiplier*(1+self.z))

        self.ha_lum = self.ha * 4 * np.pi * (cosmo.luminosity_distance(self.z).to('cm').value)**2

        goodind = np.where(np.log10(self.ha_lum) < 45)[0]

        for x, colname in enumerate(list(self.__dict__.keys())):
            if colname != 'Calzetti2000':
                setattr(self, colname, getattr(self, colname)[goodind])

        self.oiii_ew = self.oiii4959_ew + self.oiii5007_ew
        self.oii_ew = self.oii3726_ew + self.oii3729_ew
        self.oiii_ew_err = np.sqrt(self.oiii4959_ew_err**2. + self.oiii5007_ew_err**2.)
        self.oii_ew_err = np.sqrt(self.oii3726_ew_err**2. + self.oii3729_ew_err**2.)





    def Calzetti2000_novec(self, lam):

        # Plug in lam in angstroms
        # From Calzetti2000
        # Returns k(lam)

        lam = lam * 0.0001 # Convert angstroms to microns

        # Rprime_v = 4.88 # pm 0.98 from Calzetti 1997b
        Rprime_v = 4.05

        if lam > 0.1200 and lam < 0.6300:

            return 2.659 * (-2.156 + (1.509/lam) - (0.198/(lam**2.)) + (0.011/(lam**3.))) + Rprime_v

        elif lam > 0.6300 and lam < 2.2000:

            return 2.659 * (-1.857 + (1.04/lam)) + Rprime_v

        else:

            return np.NaN
