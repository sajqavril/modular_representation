"""Updated MoRe adaptation API.

This neutral entry point is the implementation used by the revised adaptation
runners.  It preserves the original MoRe separation between identification,
gating, and temporal adaptation while using the updated saved-state gate.
"""
from profile_more import (ProfileOldState, fit_profile_old_state, profile_span_gate,
                          target_aligned_old_coordinates,
                          target_profile_new_initialization)

__all__ = ['ProfileOldState', 'fit_profile_old_state', 'profile_span_gate',
           'target_aligned_old_coordinates', 'target_profile_new_initialization']
