{%- extends 'rst.tpl' -%}

{% block data_text scoped %}
.. raw:: html

    <pre class='nb-text-output'>
{{ output.text['text/plain'] | indent }}
    </pre>
{% endblock data_text %}

{% block stream %}
.. raw:: html

    <pre class='nb-text-output'>
{{ output.text['text/plain'] | indent }}
    </pre>
{% endblock stream %}
