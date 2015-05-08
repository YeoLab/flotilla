{%- extends 'rst.tpl' -%}

{% block data_text scoped %}
.. raw:: html

    <pre class='nb-text-output'>
{{ output.data['text/plain'] | indent }}
    </pre>
{% endblock data_text %}