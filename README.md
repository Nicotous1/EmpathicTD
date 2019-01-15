<!DOCTYPE html>
<html>

<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <link rel="stylesheet" href="https://stackedit.io/style.css" />
</head>

<body class="stackedit">
  <div class="stackedit__html"><h1 id="welcome-to-tdcomparator">Welcome to TDComparator</h1>
<p>This library has been made to enable you to run and compare easily Temporal-Difference algorithm on Markov Reward Process. It contains many classes to build quickly and easily your model.</p>
<p>The On and Off-TD(0) learning algorithms have been implemented. Also the new <a href="https://arxiv.org/abs/1507.01569">emphatic TD of Sutton &amp; al.</a> has been added. This library follow the works of Sutton and thus implemented the different examples found in their paper.</p>
<p>The algorithm and formula used in the library are all from the paper of Sutton &amp; al. The paper is freely available <a href="https://arxiv.org/abs/1507.01569">here</a>.</p>
<h2 id="requirements">Requirements</h2>
<p>The library has very few requirements :</p>
<ul>
<li>Python 3</li>
<li>Numpy</li>
<li>Matplotlib</li>
</ul>
<h2 id="installation">Installation</h2>
<p>To use the class of the library, you just need to import its main folder to your Python. You can do it like that :</p>
<pre class=" language-python"><code class="prism  language-python"><span class="token keyword">import</span> sys
sys<span class="token punctuation">.</span>path<span class="token punctuation">.</span>insert<span class="token punctuation">(</span><span class="token number">0</span><span class="token punctuation">,</span> <span class="token string">"library/"</span><span class="token punctuation">)</span> <span class="token comment"># Path of the library folder on your computer</span>
</code></pre>
<h2 id="documentation">Documentation</h2>
<p>To understand the library, you can look at the example in the folder “examples”.  This is a list of small tutorials :</p>
<ol>
<li><a href="https://github.com/Nicotous1/EmpathicTD/blob/master/examples/1%20-%20The%20basics.ipynb">The basics</a> : create a two states model and run the emphatic TD.</li>
<li><a href="https://github.com/Nicotous1/EmpathicTD/blob/master/examples/2%20-%20Comparing%20algorithms.ipynb">Comparing algorithms</a> : create a five states model and compare the off-TD(0) and the emphaticTD(0)</li>
<li><a href="https://github.com/Nicotous1/EmpathicTD/blob/master/examples/3%20-%20Tuning%20hyper-parameters.ipynb">Tuning hyper-parameters</a> : Quick optimization of <span class="katex--inline"><span class="katex"><span class="katex-mathml"><math><semantics><mrow><mi>α</mi></mrow><annotation encoding="application/x-tex">\alpha</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.43056em; vertical-align: 0em;"></span><span class="mord mathit" style="margin-right: 0.0037em;">α</span></span></span></span></span> and <span class="katex--inline"><span class="katex"><span class="katex-mathml"><math><semantics><mrow><mi>λ</mi></mrow><annotation encoding="application/x-tex">\lambda</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.69444em; vertical-align: 0em;"></span><span class="mord mathit">λ</span></span></span></span></span> for the emphatic-TD</li>
<li><a href="https://github.com/Nicotous1/EmpathicTD/blob/master/examples/4%20-%202D%20grid.ipynb">2D grid</a> : Create a <span class="katex--inline"><span class="katex"><span class="katex-mathml"><math><semantics><mrow><mn>5</mn><mo>×</mo><mn>5</mn></mrow><annotation encoding="application/x-tex">5 \times 5</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.72777em; vertical-align: -0.08333em;"></span><span class="mord">5</span><span class="mspace" style="margin-right: 0.222222em;"></span><span class="mbin">×</span><span class="mspace" style="margin-right: 0.222222em;"></span></span><span class="base"><span class="strut" style="height: 0.64444em; vertical-align: 0em;"></span><span class="mord">5</span></span></span></span></span> grid and run the off-TD(0) and the emphatic-TD(0)</li>
</ol>
<h2 id="files-structure">Files structure</h2>
<p>The library contains 4 files, I will briefly describe what they contain :</p>
<ul>
<li><a href="https://github.com/Nicotous1/EmpathicTD/blob/master/library/TD.py">TD.py</a> -&gt; contains all the TD algorithms (inherited from AbstractTD)
<ul>
<li>Off-TD(0)</li>
<li>Emphatic-TD from Sutton</li>
</ul>
</li>
<li><a href="https://github.com/Nicotous1/EmpathicTD/blob/master/library/policies.py">policies.py</a> -&gt; differents policies (inherited from Policy)
<ul>
<li>RightOrLeft : move right or left defined by <span class="katex--inline"><span class="katex"><span class="katex-mathml"><math><semantics><mrow><msub><mi>p</mi><mrow><mi>r</mi><mi>i</mi><mi>g</mi><mi>h</mi><mi>t</mi></mrow></msub></mrow><annotation encoding="application/x-tex">p_{right}</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.716668em; vertical-align: -0.286108em;"></span><span class="mord"><span class="mord mathit">p</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.336108em;"><span class="" style="top: -2.55em; margin-left: 0em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight"><span class="mord mathit mtight" style="margin-right: 0.02778em;">r</span><span class="mord mathit mtight">i</span><span class="mord mathit mtight" style="margin-right: 0.03588em;">g</span><span class="mord mathit mtight">h</span><span class="mord mathit mtight">t</span></span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.286108em;"><span class=""></span></span></span></span></span></span></span></span></span></span></li>
<li>GridRandomWalk : a random walk defined by <span class="katex--inline"><span class="katex"><span class="katex-mathml"><math><semantics><mrow><msub><mi>p</mi><mrow><mi>u</mi><mi>p</mi></mrow></msub><mo separator="true">,</mo><msub><mi>p</mi><mrow><mi>d</mi><mi>o</mi><mi>w</mi><mi>n</mi></mrow></msub><mo separator="true">,</mo><msub><mi>p</mi><mrow><mi>l</mi><mi>e</mi><mi>f</mi><mi>t</mi></mrow></msub><mo separator="true">,</mo><msub><mi>p</mi><mrow><mi>r</mi><mi>i</mi><mi>g</mi><mi>h</mi><mi>t</mi></mrow></msub></mrow><annotation encoding="application/x-tex">p_{up}, p_{down}, p_{left}, p_{right}</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.716668em; vertical-align: -0.286108em;"></span><span class="mord"><span class="mord mathit">p</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.151392em;"><span class="" style="top: -2.55em; margin-left: 0em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight"><span class="mord mathit mtight">u</span><span class="mord mathit mtight">p</span></span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.286108em;"><span class=""></span></span></span></span></span></span><span class="mpunct">,</span><span class="mspace" style="margin-right: 0.166667em;"></span><span class="mord"><span class="mord mathit">p</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.336108em;"><span class="" style="top: -2.55em; margin-left: 0em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight"><span class="mord mathit mtight">d</span><span class="mord mathit mtight">o</span><span class="mord mathit mtight" style="margin-right: 0.02691em;">w</span><span class="mord mathit mtight">n</span></span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.15em;"><span class=""></span></span></span></span></span></span><span class="mpunct">,</span><span class="mspace" style="margin-right: 0.166667em;"></span><span class="mord"><span class="mord mathit">p</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.336108em;"><span class="" style="top: -2.55em; margin-left: 0em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight"><span class="mord mathit mtight" style="margin-right: 0.01968em;">l</span><span class="mord mathit mtight">e</span><span class="mord mathit mtight" style="margin-right: 0.10764em;">f</span><span class="mord mathit mtight">t</span></span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.286108em;"><span class=""></span></span></span></span></span></span><span class="mpunct">,</span><span class="mspace" style="margin-right: 0.166667em;"></span><span class="mord"><span class="mord mathit">p</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.336108em;"><span class="" style="top: -2.55em; margin-left: 0em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight"><span class="mord mathit mtight" style="margin-right: 0.02778em;">r</span><span class="mord mathit mtight">i</span><span class="mord mathit mtight" style="margin-right: 0.03588em;">g</span><span class="mord mathit mtight">h</span><span class="mord mathit mtight">t</span></span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.286108em;"><span class=""></span></span></span></span></span></span></span></span></span></span>.</li>
</ul>
</li>
<li><a href="https://github.com/Nicotous1/EmpathicTD/blob/master/library/models.py">models.py</a> -&gt; contains the model to store your parameters
<ul>
<li>Model : the basic class to store your parameter.</li>
<li>Grid : A class to quickly create a grid model</li>
</ul>
</li>
<li><a href="https://github.com/Nicotous1/EmpathicTD/blob/master/library/utils.py">utils.py</a> -&gt; useful tools to analyse and paralelize the computation with numpy
<ul>
<li>comparatorTD : the tool to compute and compare the TD</li>
</ul>
</li>
</ul>
</div>
</body>

</html>
