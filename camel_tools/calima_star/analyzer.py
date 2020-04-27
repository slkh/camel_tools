# -*- coding: utf-8 -*-

# MIT License
#
# Copyright 2018-2019 New York University Abu Dhabi
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# test for branching
"""The morphological analyzer component of CALIMA Star.
"""


from __future__ import absolute_import

from collections import deque, namedtuple
import copy
import itertools
import re
from threading import RLock

from cachetools import LFUCache, cached

from camel_tools.utils.charsets import UNICODE_PUNCT_SYMBOL_CHARSET
from camel_tools.utils.charsets import AR_CHARSET, AR_DIAC_CHARSET

from camel_tools.utils.charmap import CharMapper
from camel_tools.calima_star.database import CalimaStarDB
from camel_tools.calima_star.errors import AnalyzerError
from camel_tools.calima_star.utils import merge_features
from camel_tools.calima_star.utils import simple_ar_to_caphi
from camel_tools.utils.dediac import dediac_ar


_ALL_PUNC = u''.join(UNICODE_PUNCT_SYMBOL_CHARSET)

_DIAC_RE = re.compile(u'[' + re.escape(u''.join(AR_DIAC_CHARSET)) + u']')
_IS_DIGIT_RE = re.compile(u'^.*[0-9\u0660-\u0669]+.*$')
_IS_STRICT_DIGIT_RE = re.compile(u'^[0-9\u0660-\u0669]+$')
_IS_PUNC_RE = re.compile(u'^[' + re.escape(_ALL_PUNC) + u']+$')
_HAS_PUNC_RE = re.compile(u'[' + re.escape(_ALL_PUNC) + u']')
_IS_AR_RE = re.compile(u'^[' + re.escape(u''.join(AR_CHARSET)) + u']+$')

_LEMMA_SPLIT_RE = re.compile(r'(-|_)')
# Identify No Analysis marker
_NOAN_RE = re.compile(u'NOAN')

_COPY_FEATS = frozenset(['gloss', 'atbtok', 'atbseg', 'd1tok', 'd1seg',
                         'd2tok', 'd2seg', 'd3tok', 'd3seg'])

_UNDEFINED_LEX_FEATS = frozenset(['root', 'pattern', 'caphi'])

DEFAULT_NORMALIZE_MAP = CharMapper({
    u'\u0625': u'\u0627',
    u'\u0623': u'\u0627',
    u'\u0622': u'\u0627',
    u'\u0671': u'\u0627',
    u'\u0649': u'\u064a',
    u'\u0629': u'\u0647',
    u'\u0640': u''
})
""":obj:`~camel_tools.utils.charmap.CharMapper`: The default character map used
for normalization by :obj:`CalimaStarAnalyzer`.

Removes the tatweel/kashida character and does the following conversions:

- 'إ' to 'ا'
- 'أ' to 'ا'
- 'آ' to 'ا'
- 'ٱ' to 'ا'
- 'ى' to 'ي'
- 'ة' to 'ه'
"""


_BACKOFF_TYPES = frozenset(['NONE', 'NOAN_ALL', 'NOAN_PROP', 'ADD_ALL',
                            'ADD_PROP', 'ADD_PATT', 'NOAN_PATT'])


class AnalyzedWord(namedtuple('AnalyzedWord', ['word', 'analyses'])):
    """A named tuple containing a word and its analyses.

    Attributes:
        word (:obj:`str`): The analyzed word.

        analyses (:obj:`list` of :obj:`dict`): List of analyses for **word**.
            See :doc:`/reference/calima_star_features` for more information on
            features and their values.
    """


def _is_digit(word):
    return _IS_DIGIT_RE.match(word) is not None


def _is_strict_digit(word):
    return _IS_STRICT_DIGIT_RE.match(word) is not None


def _is_punc(word):
    return _IS_PUNC_RE.match(word) is not None


def _has_punc(word):
    return _HAS_PUNC_RE.search(word) is not None


def _is_ar(word):
    return _IS_AR_RE.match(word) is not None


def _segments_gen(word, max_prefix=1, max_suffix=1):
    w = len(word)
    for p in range(0, min(max_prefix, w - 1) + 1):
        prefix = word[:p]
        for s in range(max(1, w - p - max_suffix), w - p + 1):
            stem = word[p:p+s]
            suffix = word[p+s:]
            yield (prefix, stem, suffix)


class CalimaStarAnalyzer:
    """CALIMA Star analyzer component.

    Args:
        db (:obj:`~camel_tools.calima_star.database.CalimaStarDB`): Database to
            use for analysis. Must be opened in analysis or reinflection mode.
        backoff (:obj:`str`, optional): Backoff mode. Can be one of the
            following: 'NONE', 'NOAN_ALL', 'NOAN_PROP', 'ADD_ALL', or
            'ADD_PROP'. Defaults to 'NONE'.
        norm_map (:obj:`~camel_tools.utils.charmap.CharMapper`, optional):
            Character map for normalizing input words. Defaults to
            :const:`DEFAULT_NORMALIZE_MAP`.
        strict_digit (:obj:`bool`, optional): If set to `True`, then only words
            completely comprised of digits are considered numbers, otherwise,
            all words containing a digit are considered numbers. Defaults to
            `False`.
        cache_size (:obj:`int`, optional): If greater than zero, then the
            analyzer will cache the analyses for the **cache_Size** most
            frequent words, otherwise no analyses will be cached.

    Raises:
        :obj:`~camel_tools.calima_star.errors.AnalyzerError`: If database is
            not an instance of
            (:obj:`~camel_tools.calima_star.database.CalimaStarDB`), if **db**
            does not support analysis, or if **backoff** is not a valid backoff
            mode.
    """

    def __init__(self, db, backoff='NONE',
                 norm_map=DEFAULT_NORMALIZE_MAP,
                 strict_digit=False,
                 cache_size=0):
        if not isinstance(db, CalimaStarDB):
            raise AnalyzerError('DB is not an instance of CalimaStarDB')
        if not db.flags.analysis:
            raise AnalyzerError('DB does not support analysis')

        self._db = db

        self._backoff = backoff
        self._norm_map = DEFAULT_NORMALIZE_MAP
        self._strict_digit = strict_digit

        if backoff in _BACKOFF_TYPES:
            if backoff == 'NONE':
                self._backoff_condition = None
                self._backoff_action = None
            else:
                backoff_toks = backoff.split('_')
                self._backoff_condition = backoff_toks[0]
                self._backoff_action = backoff_toks[1]
        else:
            raise AnalyzerError('Invalid backoff mode {}'.format(
                repr(backoff)))

        if isinstance(cache_size, int):
            if cache_size > 0:
                cache = LFUCache(cache_size)
                self.analyze = cached(cache, lock=RLock())(self.analyze)

        else:
            raise AnalyzerError('Invalid cache size {}'.format(
                                repr(cache_size)))

    def _normalize(self, word):
        if self._norm_map is None:
            return word
        return self._norm_map.map_string(word)

    def _combined_analyses(self,
                           word_dediac,
                           prefix_analyses,
                           stem_analyses,
                           suffix_analyses):
        combined = deque()

        for p in itertools.product(prefix_analyses, stem_analyses):
            prefix_cat = p[0][0]
            prefix_feats = p[0][1]
            stem_cat = p[1][0]
            stem_feats = p[1][1]

            if stem_cat in self._db.prefix_stem_compat[prefix_cat]:
                for suffix_cat, suffix_feats in suffix_analyses:
                    if ((stem_cat not in self._db.stem_suffix_compat) or
                        (prefix_cat not in self._db.prefix_suffix_compat) or
                        (suffix_cat not in
                         self._db.stem_suffix_compat[stem_cat]) or
                        (suffix_cat not in
                         self._db.prefix_suffix_compat[prefix_cat])):
                        continue

                    merged = merge_features(self._db, prefix_feats, stem_feats,
                                            suffix_feats)
                    merged['stem'] = stem_feats['diac']
                    merged['stemcat'] = stem_cat

                    merged_dediac = dediac_ar(merged['diac'])
                    if word_dediac.replace(u'\u0640', '') != merged_dediac:
                        merged['source'] = 'spvar'

                    # override the generated diacritized text with the word itself (for dev reasons only), remove cat from lex
                    #TODO: remove override once backoff is official
                    merged['diac'] = dediac_ar(merged['diac'])
                    # merged['lex'] = _LEMMA_SPLIT_RE.split(merged['lex'])[0]
                    combined.append(merged)

        return combined

    def _match_word(self, surf, surf_patt, stem_patt):
        for i in range(0, len(surf)):
            if (surf_patt[i] != stem_patt [i]) and (surf[i] != stem_patt[i]):
                return False
        return True
    def _get_stem_anls(self, surf, surf_patt):
        anls = []
        for stem_patt in self._db.stem_patt_hash:
            if len(stem_patt) != len(surf):
                continue
            if stem_patt == surf_patt:
                anls.extend(self._db.stem_patt_hash[stem_patt])
            elif self._match_word(surf, surf_patt, stem_patt):
                anls.extend(self._db.stem_patt_hash[stem_patt])
            else:
                continue
        if anls:
            return anls
        else:
            return None

    def _combined_backoff_analyses(self,
                                   stem,
                                   word_dediac,
                                   prefix_analyses,
                                   stem_analyses,
                                   suffix_analyses):
        combined = deque()

        for p in itertools.product(prefix_analyses, stem_analyses):
            prefix_cat = p[0][0]
            prefix_feats = p[0][1]
            stem_cat = p[1][0]
            stem_feats = copy.copy(p[1][1])

            if stem_cat in self._db.prefix_stem_compat[prefix_cat]:
                for suffix_cat, suffix_feats in suffix_analyses:
                    if ((suffix_cat not in
                         self._db.stem_suffix_compat[stem_cat]) or
                        (prefix_cat not in self._db.prefix_suffix_compat or
                         suffix_cat not in
                         self._db.prefix_suffix_compat[prefix_cat])):
                        continue
                    if 'bw' in stem_feats:
                        if (self._backoff_action == 'PROP' and
                            'NOUN_PROP' not in stem_feats['bw']):
                            continue

                        stem_feats['bw'] = _NOAN_RE.sub(stem, stem_feats['bw'])
                    stem_feats['diac'] = _NOAN_RE.sub(stem, stem_feats['diac'])
                    stem_feats['lex'] = _NOAN_RE.sub(stem, stem_feats['lex'])
                    if 'caphi' in stem_feats:
                        stem_feats['caphi'] = simple_ar_to_caphi(stem)

                    merged = merge_features(self._db, prefix_feats, stem_feats,
                                            suffix_feats)

                    merged['stem'] = stem_feats['diac']
                    merged['stemcat'] = stem_cat
                    merged['source'] = 'backoff'
                    if 'gloss' in stem_feats:
                        merged['gloss'] = stem_feats['gloss']
                    # override the generated diacritized text with the word itself (for dev reasons only), remove cat from lex
                    #TODO: remove override once backoff is official
                    merged['diac'] = dediac_ar(merged['diac'])
                    # merged['lex'] = _LEMMA_SPLIT_RE.split(merged['lex'])[0]
                    combined.append(merged)

        return combined

    def _combined_patt_backoff_analyses(self,
                                   stem, surf_patt,
                                   orth_root, word_dediac,
                                   prefix_analyses,
                                   stem_analyses,
                                   suffix_analyses):
        combined = deque()
        for p in itertools.product(prefix_analyses, stem_analyses):
            prefix_cat = p[0][0]
            prefix_feats = p[0][1]
            stem_cat = p[1][0]
            stem_feats = copy.copy(p[1][1])
            if stem_cat in self._db.prefix_stem_compat[prefix_cat]:
                for suffix_cat, suffix_feats in suffix_analyses:
                    if ((suffix_cat not in
                         self._db.stem_suffix_compat[stem_cat]) or
                        (prefix_cat not in self._db.prefix_suffix_compat or
                         suffix_cat not in
                         self._db.prefix_suffix_compat[prefix_cat])):
                        continue

                    # if (self._backoff_action == 'PROP' and
                    #         'NOUN_PROP' not in stem_feats['bw']):
                    #     continue

                    ## surface patt to regex
                    stem_patt = re.sub('\d', '(.)', surf_patt)
                    root_patt = re.sub('.', r'(.)', orth_root)
                    diac_patt_regex = re.sub('(\d)', r'\\\g<1>', stem_feats['diac'])
                    # d3tok_patt_regex = re.sub('(\d)', r'\\\g<1>', stem_feats['d3tok'])
                    
                    
                    #stem_feats['d3tok'] = re.sub(stem_patt, d3tok_patt_regex, stem)
                    # stem_feats['diac'] = re.sub(stem_patt, diac_patt_regex, stem)
                    stem_feats['diac'] = re.sub(root_patt, diac_patt_regex, orth_root)
                    # lemma = _LEMMA_SPLIT_RE.split(stem_feats['lex'])[0]
                    # sep = _LEMMA_SPLIT_RE.split(stem_feats['lex'])[1]
                    # cat = _LEMMA_SPLIT_RE.split(stem_feats['lex'])[2]
                    lex_patt_regex = re.sub('(\d)', r'\\\g<1>', stem_feats['lex'])
                    # stem_feats['lex'] = '{}{}'.format(re.sub(stem_patt, lex_patt_regex, stem), ''.join(_LEMMA_SPLIT_RE.split(stem_feats['lex'])[1:]))
                    # stem_feats['lex'] = re.sub(stem_patt, lex_patt_regex, stem)
                    
                    stem_feats['lex'] = re.sub(root_patt, lex_patt_regex, orth_root)
                    # print('me here')
                    ## BW, mush special such wow, so annoying but so useful
                    
                    # bw_elements = stem_feats['bw'].split('+')
                    # new_bw = []
                    # for elem in bw_elements:
                    #     elem_lex = elem.split('/')[0]
                    #     elem_pos = elem.split('/')[1]

                    #     elem_lex_patt = re.sub('(\d)', r'\\\g<1>', elem_lex)
                    #     elem_lex = re.sub(stem_patt, elem_lex_patt, stem)
                        
                    #     elem = '{}/{}'.format(elem_lex, elem_pos)
                    #     new_bw.append(elem)
                    
                    # stem_feats['bw'] = '+'.join(new_bw)
                    merged = merge_features(self._db, prefix_feats, stem_feats,
                                            suffix_feats)
                    # override the generated diacritized text with the word itself (for dev reasons only), remove cat from lex
                    #TODO: remove override once backoff is official
                    merged['diac'] = dediac_ar(merged['diac'])
                    # merged['lex'] = _LEMMA_SPLIT_RE.split(merged['lex'])[0]

                    ####
                    merged['stem'] = stem_feats['diac']
                    merged['stemcat'] = stem_cat
                    merged['source'] = 'patt_backoff'
                    # merged['gloss'] = stem_feats['gloss']

                    combined.append(merged)
                    # print(combined)
        return combined

    def analyze(self, word):
        """Analyze a given word.

        Args:
            word (:py:obj:`str`): Word to analyze.

        Returns:
            :obj:`list` of :obj:`dict`: The list of analyses for **word**.
            See :doc:`/reference/calima_star_features` for more information on
            features and their values.
        """

        word = word.strip()

        if word == '':
            return []

        analyses = deque()
        word_dediac = dediac_ar(word)
        word_normal = self._normalize(word_dediac)

        if ((self._strict_digit and _is_strict_digit(word)) or
                (not self._strict_digit and _is_digit(word))):
            result = copy.copy(self._db.defaults['digit'])
            result['diac'] = word
            result['stem'] = word
            result['stemgloss'] = word
            result['stemcat'] = None
            result['lex'] = word + '_0'
            result['bw'] = word + '/NOUN_NUM'
            result['source'] = 'digit'

            for feat in _COPY_FEATS:
                if feat in self._db.defines:
                    result[feat] = word

            for feat in _UNDEFINED_LEX_FEATS:
                if feat in self._db.defines:
                    result[feat] = 'DIGIT'

            if 'catib6' in self._db.defines:
                result['catib6'] = 'NOM'
            if 'ud' in self._db.defines:
                result['ud'] = 'NUM'

            result['pos_freq'] = -99.0
            result['lex_freq'] = -99.0
            result['pos_lex_freq'] = -99.0

            return [result]

        elif _is_punc(word):
            result = copy.copy(self._db.defaults['punc'])
            result['diac'] = word
            result['stem'] = word
            result['stemgloss'] = word
            result['stemcat'] = None
            result['lex'] = word + '_0'
            result['bw'] = word + '/PUNC'
            result['source'] = 'punc'

            for feat in _COPY_FEATS:
                if feat in self._db.defines:
                    result[feat] = word

            for feat in _UNDEFINED_LEX_FEATS:
                if feat in self._db.defines:
                    result[feat] = 'PUNC'

            if 'catib6' in self._db.defines:
                result['catib6'] = 'PNX'
            if 'ud' in self._db.defines:
                result['ud'] = 'PUNCT'

            result['pos_freq'] = -99.0
            result['lex_freq'] = -99.0
            result['pos_lex_freq'] = -99.0

            return [result]

        elif _has_punc(word):
            pass

        elif not _is_ar(word):
            result = copy.copy(self._db.defaults['noun'])
            result['diac'] = word
            result['stem'] = word
            result['stemgloss'] = word
            result['stemcat'] = None
            result['lex'] = word + '_0'
            result['bw'] = word + '/FOREIGN'
            result['source'] = 'foreign'

            for feat in _COPY_FEATS:
                if feat in self._db.defines:
                    result[feat] = word

            for feat in _UNDEFINED_LEX_FEATS:
                if feat in self._db.defines:
                    result[feat] = 'FOREIGN'

            if 'catib6' in self._db.defines:
                result['catib6'] = 'FOREIGN'

            if 'ud' in self._db.defines:
                result['ud'] = 'X'

            result['pos_freq'] = -99.0
            result['lex_freq'] = -99.0
            result['pos_lex_freq'] = -99.0

            return [result]

        else:
            segments_gen = _segments_gen(word_normal, self._db.max_prefix_size,
                                         self._db.max_suffix_size)

            for segmentation in segments_gen:
                prefix = segmentation[0]
                stem = segmentation[1]
                suffix = segmentation[2]

                prefix_analyses = self._db.prefix_hash.get(prefix, None)
                suffix_analyses = self._db.suffix_hash.get(suffix, None)

                if prefix_analyses is None or suffix_analyses is None:
                    continue

                stem_analyses = self._db.stem_hash.get(stem, None)

                if stem_analyses is not None:
                    combined = self._combined_analyses(word_dediac,
                                                       prefix_analyses,
                                                       stem_analyses,
                                                       suffix_analyses)
                    analyses.extend(combined)

        if ((self._backoff_condition == 'NOAN' and len(analyses) == 0) or
                (self._backoff_condition == 'ADD')) and self._backoff_action != 'PATT':
            segments_gen = _segments_gen(word_normal,
                                         self._db.max_prefix_size,
                                         self._db.max_suffix_size)

            backoff_cats = self._db.stem_backoffs[self._backoff_action]
            stem_analyses = [(cat, analysis)
                             for cat, analysis in self._db.stem_hash['NOAN']
                             if cat in backoff_cats]

            for segmentation in segments_gen:
                prefix = segmentation[0]
                stem = segmentation[1]
                suffix = segmentation[2]

                prefix_analyses = self._db.prefix_hash.get(prefix, None)
                suffix_analyses = self._db.suffix_hash.get(suffix, None)

                if prefix_analyses is None or suffix_analyses is None:
                    continue

                combined = self._combined_backoff_analyses(stem,
                                                           word_dediac,
                                                           prefix_analyses,
                                                           stem_analyses,
                                                           suffix_analyses)
                analyses.extend(combined)
        if ((self._backoff_action == 'PATT' and len(analyses) == 0) or
                (self._backoff_condition == 'ADD') and (self._backoff_action == 'PATT')):
            segments_gen = _segments_gen(word_normal,
                                        self._db.max_prefix_size,
                                        self._db.max_suffix_size)

        
            # backoff_cats = self._db.stem_backoffs[self._backoff_action]
            # stem_analyses = [(cat, analysis)
            #                  for cat, analysis in self._db.stem_hash['NOAN']
            #                  if cat in backoff_cats]
            
            for segmentation in segments_gen:
                # print(segmentation)
                stem_analyses = []
                prefix = segmentation[0]
                stem = segmentation[1]
                suffix = segmentation[2]

                prefix_analyses = self._db.prefix_hash.get(prefix, None)
                suffix_analyses = self._db.suffix_hash.get(suffix, None)

                ## get stem root radicals, where Ayw are considered part of the pattren
                orth_root = ''.join(sorted(set(stem) & 
                                set(stem), key = stem.index))

                orth_root = re.sub('[أاويىآإؤئء]', '', orth_root)
                ## get the surface pattren with numerals
                surf_patt = stem
                for char in orth_root:
                    surf_patt = re.sub(char, str(orth_root.index(char)+1), surf_patt)
                # get the stem analyses using the special hash based on the stem patt from the db
                for key in self._db.stem_patt_hash:
                    general_patt = key.split('-')[1]
                    numeric_patt = key.split('-')[0]
                    if re.fullmatch(general_patt, stem) is not None:
                        if surf_patt == numeric_patt:
                            # print(surf_patt, numeric_patt, stem, key)
                            stem_analyses.extend(self._db.stem_patt_hash.get(key))
                
                if not stem_analyses:
                    # print('HEY')
                    stem_analyses = None
                # stem_analyses = self._db.stem_patt_hash.get(surf_patt, None)
                if stem_analyses is None:
                    continue
                if prefix_analyses is None or suffix_analyses is None:
                    continue
                ## if no pattern match, revert to basic NOAN_PROP backoff
                # print(stem_analyses is None)
                ### Add the NOAN_PROP options anyways for now ####################################
                # if not stem_analyses:
                # backoff_cats = self._db.stem_backoffs['PROP']
                # print(backoff_cats)
                
                # stem_analyses = [(cat, analysis)
                #                 for cat, analysis in self._db.stem_hash['NOAN']
                #                 if cat in backoff_cats]
                # # print('me here', stem_analyses)
                # combined = self._combined_backoff_analyses(stem,
                #                                         word_dediac,
                #                                         prefix_analyses,
                #                                         stem_analyses,
                #                                         suffix_analyses)
                # analyses.extend(combined)
                
                    # print(combined)
                ###################################################################################
                # else:
                # stem_analyses = self._db.stem_patt_hash.get(surf_patt, None)
                # if not stem_analyses:
                #     continue
                # print(stem_analyses)
                combined = self._combined_patt_backoff_analyses(stem, surf_patt,
                                                            orth_root, word_dediac,
                                                            prefix_analyses,
                                                            stem_analyses,
                                                            suffix_analyses)
                
                analyses.extend(combined)
            # print('me here', stem)
        result = list(analyses)

        return result

    def analyze_words(self, words):
        '''Analyze a list of words.

        Args:
            words (:py:obj:`list` of :py:obj:`str`): List of words to analyze.

        Returns:
            :obj:`list` of :obj:`AnalyzedWord`: The list of analyses for each
            word in **words**.
        '''

        return list(map(lambda w: AnalyzedWord(w, self.analyze(w)), words))
