{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

{% if name in ["Node", "ProxyNode", "MultilinearNode", "Classical"] %}
.. autoclass:: {{ objname }}
   :show-inheritance:
   :members:
   :inherited-members:
   :exclude-members: _abc_impl
   :undoc-members:
   :private-members:
   :member-order: groupwise
{% elif "_subspace_in" in members %}
.. autoclass:: {{ objname }}
   :show-inheritance:
{% else %}
.. autoclass:: {{ objname }}
   :show-inheritance:
   :members:
   :inherited-members:
   :undoc-members:
   :member-order: groupwise
{% endif %}
